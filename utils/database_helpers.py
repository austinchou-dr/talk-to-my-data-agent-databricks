# Copyright 2024 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import functools
import json
import traceback
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Generic, TypeVar, cast, List, Optional # UPDATED FOR DATABRICKS

import pandas as pd
import polars as pl
import snowflake.connector
from google.cloud import bigquery
from hdbcli import dbapi
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from pydantic import ValidationError
from databricks.sdk.core import Config, oauth_service_principal # ADDED FOR DATABRICKS
from databricks import sql  # ADDED FOR DATABRICKS

from utils.analyst_db import AnalystDB, DataSourceType
from utils.code_execution import InvalidGeneratedCode
from utils.credentials import (
    GoogleCredentialsBQ,
    NoDatabaseCredentials,
    SAPDatasphereCredentials,
    SnowflakeCredentials,
    DataBricksCredentials # ADDED FOR DATABRICKS
)
from utils.logging_helper import get_logger
from utils.prompts import (
    SYSTEM_PROMPT_BIGQUERY,
    SYSTEM_PROMPT_SAP_DATASPHERE,
    SYSTEM_PROMPT_SNOWFLAKE,
    SYSTEM_PROMPT_DATABRICKS # ADDED FOR DATABRICKS
)
from utils.schema import (
    AnalystDataset,
    AppInfra,
)

logger = get_logger("DatabaseHelper")

T = TypeVar("T")
_DEFAULT_DB_QUERY_TIMEOUT = 300


@dataclass
class SnowflakeCredentialArgs:
    credentials: SnowflakeCredentials


@dataclass
class BigQueryCredentialArgs:
    credentials: GoogleCredentialsBQ


@dataclass
class SAPDatasphereCredentialArgs:
    credentials: SAPDatasphereCredentials


@dataclass
class NoDatabaseCredentialArgs:
    credentials: NoDatabaseCredentials


@dataclass
class DataBricksCredentialArgs: # ADDED FOR DATABRICKS
    credentials: DataBricksCredentials


class DatabaseOperator(ABC, Generic[T]):
    @abstractmethod
    def __init__(self, credentials: T, default_timeout: int): ...

    @abstractmethod
    @contextmanager
    def create_connection(self) -> Any: ...

    @abstractmethod
    def execute_query(
        self, query: str, timeout: int | None = None
    ) -> list[tuple[Any, ...]] | list[dict[str, Any]]: ...

    @abstractmethod
    def get_tables(self, timeout: int | None = None) -> list[str]:
        return []

    @functools.lru_cache(maxsize=8)
    @abstractmethod
    async def get_data(
        self,
        *table_names: str,
        analyst_db: AnalystDB,
        sample_size: int = 5000,
        timeout: int | None = None,
    ) -> list[str]:
        return []

    @abstractmethod
    def get_system_prompt(self) -> ChatCompletionSystemMessageParam:
        return ChatCompletionSystemMessageParam(role="system", content="")


class NoDatabaseOperator(DatabaseOperator[NoDatabaseCredentialArgs]):
    def __init__(
        self,
        credentials: NoDatabaseCredentials,
        default_timeout: int = _DEFAULT_DB_QUERY_TIMEOUT,
    ):
        self._credentials = credentials

    @contextmanager
    def create_connection(self) -> Generator[None]:
        yield None

    def execute_query(
        self,
        query: str,
        timeout: int | None = 300,
    ) -> list[tuple[Any, ...]] | list[dict[str, Any]]:
        return []

    def get_tables(self, timeout: int | None = 300) -> list[str]:
        return []

    @functools.lru_cache(8)
    async def get_data(
        self,
        *table_names: str,
        analyst_db: AnalystDB,
        sample_size: int = 5000,
        timeout: int | None = 300,
    ) -> list[str]:
        return []

    def get_system_prompt(self) -> ChatCompletionSystemMessageParam:
        return ChatCompletionSystemMessageParam(role="system", content="")


class SnowflakeOperator(DatabaseOperator[SnowflakeCredentialArgs]):
    def __init__(
        self,
        credentials: SnowflakeCredentials,
        default_timeout: int = _DEFAULT_DB_QUERY_TIMEOUT,
    ):
        if not credentials.is_configured():
            raise ValueError("Snowflake credentials not properly configured")
        self._credentials = credentials
        self.default_timeout = default_timeout

    @contextmanager
    def create_connection(self) -> Generator[snowflake.connector.SnowflakeConnection]:
        """Create a connection to Snowflake using environment variables"""
        if not self._credentials.is_configured():
            raise ValueError("Snowflake credentials not properly configured")

        connect_params: dict[str, Any] = {
            "user": self._credentials.user,
            "account": self._credentials.account,
            "warehouse": self._credentials.warehouse,
            "database": self._credentials.database,
            "schema": self._credentials.db_schema,
            "role": self._credentials.role,
        }

        # Try key file authentication first if configured
        project_root = Path(__file__).resolve().parent.parent
        if private_key := self._credentials.get_private_key(project_root=project_root):
            connect_params["private_key"] = private_key
        elif self._credentials.password:
            connect_params["password"] = self._credentials.password
        else:
            raise ValueError(
                "Neither private key nor password authentication configured"
            )

        connection = snowflake.connector.connect(**connect_params)
        yield connection
        connection.close()

    def execute_query(
        self, query: str, timeout: int | None = None
    ) -> list[tuple[Any, ...]] | list[dict[str, Any]]:
        """Execute a Snowflake query with timeout and metadata capture

        Args:
            conn: Snowflake connection
            query: SQL query to execute
            timeout: Query timeout in seconds

        Returns:
            Tuple of (results, metadata)
        """
        timeout = timeout if timeout is not None else self.default_timeout
        conn: snowflake.connector.SnowflakeConnection
        try:
            with self.create_connection() as conn:
                with conn.cursor(snowflake.connector.DictCursor) as cursor:
                    cursor = conn.cursor(snowflake.connector.DictCursor)
                    # Set query timeout at cursor level
                    cursor.execute(
                        f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {timeout}"
                    )

                    try:
                        # Execute query
                        cursor.execute(query)

                        # Get results
                        results = cursor.fetchall()

                        return results

                    except snowflake.connector.errors.ProgrammingError as e:
                        # Handle Snowflake-specific errors
                        raise InvalidGeneratedCode(
                            f"Snowflake error: {str(e.msg)}",
                            code=query,
                            exception=None,
                            traceback_str="",
                        )

        except Exception as e:
            raise InvalidGeneratedCode(
                f"Query execution failed: {str(e)}",
                code=query,
                exception=e,
                traceback_str=traceback.format_exc(),
            )

    def get_tables(self, timeout: int | None = None) -> list[str]:
        """Fetch list of tables from Snowflake schema"""
        timeout = timeout if timeout is not None else self.default_timeout

        conn: snowflake.connector.SnowflakeConnection
        try:
            with self.create_connection() as conn:
                with conn.cursor() as cursor:
                    # Log current session info
                    logger.info("Checking current session settings...")
                    cursor.execute(
                        f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {timeout}"
                    )

                    cursor.execute(
                        "SELECT CURRENT_DATABASE(), CURRENT_SCHEMA(), CURRENT_ROLE(), CURRENT_WAREHOUSE()",
                    )
                    current_settings = cursor.fetchone()
                    logger.info(
                        f"Current settings - Database: {current_settings[0]}, Schema: {current_settings[1]}, Role: {current_settings[2]}, Warehouse: {current_settings[3]}"  # type: ignore[index]
                    )

                    # Check if schema exists
                    cursor.execute(
                        f"""
                        SELECT COUNT(*)
                        FROM {self._credentials.database}.INFORMATION_SCHEMA.SCHEMATA
                        WHERE SCHEMA_NAME = '{self._credentials.db_schema}'
                    """,
                    )
                    schema_exists = cursor.fetchone()[0]  # type: ignore[index]
                    logger.info(f"Schema exists check: {schema_exists > 0}")

                    # Get all objects (tables and views)
                    cursor.execute(
                        f"""
                        SELECT table_name, table_type
                        FROM {self._credentials.database}.information_schema.tables
                        WHERE table_schema = '{self._credentials.db_schema}'
                        AND table_type IN ('BASE TABLE', 'VIEW')
                        ORDER BY table_type, table_name
                    """
                    )
                    results = cursor.fetchall()
                    tables = [row[0] for row in results]

                    # Log detailed results
                    logger.info(f"Total objects found: {len(results)}")
                    for table_name, table_type in results:
                        logger.info(f"Found {table_type}: {table_name}")

                    # Check schema privileges
                    cursor.execute(
                        f"""
                        SHOW GRANTS ON SCHEMA {self._credentials.database}.{self._credentials.db_schema}
                    """
                    )
                    privileges = cursor.fetchall()
                    logger.info("Schema privileges:")
                    for priv in privileges:
                        logger.info(f"Privilege: {priv}")

                    return tables

        except Exception as e:
            logger.error(f"Failed to fetch tables: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {str(e)}")
            return []

    @functools.lru_cache(maxsize=8)
    async def get_data(
        self,
        *table_names: str,
        analyst_db: AnalystDB,
        sample_size: int = 5000,
        timeout: int | None = None,
    ) -> list[str]:
        """Load selected tables from Snowflake as pandas DataFrames

        Args:
        - table_names: List of table names to fetch
        - sample_size: Number of rows to sample from each table

        Returns:
        - Dictionary of table names to list of records
        """

        timeout = timeout if timeout is not None else self.default_timeout

        conn: snowflake.connector.SnowflakeConnection

        dataframes = []
        try:
            with self.create_connection() as conn:
                cursor = conn.cursor()

                for table in table_names:
                    try:
                        qualified_table = f'{self._credentials.database}.{self._credentials.db_schema}."{table}"'
                        logger.info(f"Fetching data from table: {qualified_table}")
                        cursor.execute(
                            f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {timeout}"
                        )
                        cursor.execute(
                            f"""
                            SELECT * FROM {qualified_table}
                            SAMPLE ({sample_size} ROWS)
                        """
                        )

                        columns = [desc[0] for desc in cursor.description]
                        data = cursor.fetchall()
                        pandas_df = pd.DataFrame(data=data, columns=columns, dtype=str)
                        df = pl.DataFrame(
                            data=pandas_df, schema={col: pl.String for col in columns}
                        )

                        logger.info(
                            f"Successfully loaded table {table}: {len(df)} rows, {len(df.columns)} columns"
                        )
                        dataframes.append(AnalystDataset(name=table, data=df))

                    except Exception as e:
                        logger.error(f"Error loading table {table}: {str(e)}")
                        logger.error(f"Error type: {type(e)}")
                        logger.error(f"Error details: {str(e)}")
                        continue
                names = []
                for dataframe in dataframes:
                    await analyst_db.register_dataset(
                        dataframe, DataSourceType.DATABASE
                    )
                    names.append(dataframe.name)
                return names

        except Exception as e:
            logger.error(f"Error fetching Snowflake data: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {str(e)}")
            return []

    def get_system_prompt(self) -> ChatCompletionSystemMessageParam:
        return ChatCompletionSystemMessageParam(
            role="system",
            content=SYSTEM_PROMPT_SNOWFLAKE.format(
                warehouse=self._credentials.warehouse,
                database=self._credentials.database,
                schema=self._credentials.db_schema,
            ),
        )


class BigQueryOperator(DatabaseOperator[BigQueryCredentialArgs]):
    def __init__(
        self,
        credentials: GoogleCredentialsBQ,
        default_timeout: int = _DEFAULT_DB_QUERY_TIMEOUT,
    ):
        self._credentials = credentials
        self._credentials.db_schema = self._credentials.db_schema
        self._database = credentials.service_account_key["project_id"]
        self.default_timeout = default_timeout

    @contextmanager
    def create_connection(self) -> Generator[bigquery.Client]:
        from google.oauth2 import service_account

        google_credentials = service_account.Credentials.from_service_account_info(  # type: ignore[no-untyped-call]
            GoogleCredentialsBQ().service_account_key,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        client = bigquery.Client(
            credentials=google_credentials,
        )

        yield client

        client.close()  # type: ignore[no-untyped-call]

    def execute_query(
        self, query: str, timeout: int | None = None
    ) -> list[tuple[Any, ...]] | list[dict[str, Any]]:
        conn: bigquery.Client
        timeout = timeout if timeout is not None else self.default_timeout
        try:
            with self.create_connection() as conn:
                results = conn.query(query, timeout=timeout)

                sql_result: pd.DataFrame = results.to_dataframe()

                sql_result_as_dicts = cast(
                    list[dict[str, Any]], sql_result.to_dict(orient="records")
                )
                return sql_result_as_dicts

        except Exception as e:
            raise InvalidGeneratedCode(
                f"Query execution failed: {str(e)}",
                code=query,
                exception=e,
                traceback_str=traceback.format_exc(),
            )

    def get_tables(self, timeout: int | None = None) -> list[str]:
        """Fetch list of tables from BigQuery schema"""
        timeout = timeout if timeout is not None else self.default_timeout

        conn: bigquery.Client

        try:
            with self.create_connection() as conn:
                tables = [
                    i.table_id
                    for i in conn.list_tables(
                        str(self._credentials.db_schema), timeout=timeout
                    )
                ]

                # Log detailed results
                logger.info(f"Total objects found: {len(tables)}")
                logger.info(f"Found tables: {', '.join(tables)}")

                return tables

        except Exception as e:
            logger.error(f"Failed to fetch tables: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {str(e)}")
            return []

    @functools.lru_cache(maxsize=8)
    async def get_data(
        self,
        *table_names: str,
        analyst_db: AnalystDB,
        sample_size: int = 5000,
        timeout: int | None = None,
    ) -> list[str]:
        timeout = timeout if timeout is not None else self.default_timeout

        dataframes = []

        conn: bigquery.Client

        try:
            with self.create_connection() as conn:
                for table in table_names:
                    try:
                        qualified_table = (
                            f"{self._database}.{self._credentials.db_schema}.{table}"
                        )
                        logger.info(f"Fetching data from table: {qualified_table}")

                        pandas_df: pd.DataFrame = conn.query(
                            f"""
                            SELECT * FROM `{qualified_table}`
                            LIMIT {sample_size}
                        """,
                            timeout=timeout,
                        ).to_dataframe()
                        df = pl.DataFrame(
                            data=pandas_df,
                            schema={col: pl.String for col in pandas_df.columns},
                        )
                        logger.info(
                            f"Successfully loaded table {table}: {len(df)} rows, {len(df.columns)} columns"
                        )

                        dataframes.append(AnalystDataset(name=table, data=df))

                    except Exception as e:
                        logger.error(f"Error loading table {table}: {str(e)}")
                        logger.error(f"Error type: {type(e)}")
                        logger.error(f"Error details: {str(e)}")
                        continue

                names = []
                for dataframe in dataframes:
                    await analyst_db.register_dataset(
                        dataframe, DataSourceType.DATABASE
                    )
                    names.append(dataframe.name)

                return names

        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {str(e)}")
            return []

    def get_system_prompt(self) -> ChatCompletionSystemMessageParam:
        return ChatCompletionSystemMessageParam(
            role="system",
            content=SYSTEM_PROMPT_BIGQUERY.format(
                project=self._database,
                dataset=self._credentials.db_schema,
            ),
        )


class SAPDatasphereOperator(DatabaseOperator[SAPDatasphereCredentialArgs]):
    def __init__(
        self,
        credentials: SAPDatasphereCredentials,
        default_timeout: int = _DEFAULT_DB_QUERY_TIMEOUT,
    ):
        if not credentials.is_configured():
            raise ValueError("SAP Data Sphere credentials not properly configured")
        self._credentials = credentials
        self.default_timeout = default_timeout

    @contextmanager
    def create_connection(self) -> Generator[dbapi.Connection]:
        """Create a connection to SAP Data Sphere"""
        if not self._credentials.is_configured():
            raise ValueError("SAP Data Sphere credentials not properly configured")

        connect_params: dict[str, Any] = {
            "address": self._credentials.host,
            "port": self._credentials.port,
            "user": self._credentials.user,
            "password": self._credentials.password,
        }

        try:
            # Connect to SAP Data Sphere
            connection = dbapi.connect(**connect_params)
            yield connection
        except Exception:
            raise
        finally:
            connection.close()

    def execute_query(
        self, query: str, timeout: int | None = None
    ) -> list[tuple[Any, ...]] | list[dict[str, Any]]:
        """Execute a SAP Data Sphere query with timeout

        Args:
            query: SQL query to execute
            timeout: Query timeout in seconds

        Returns:
            Query results
        """
        timeout = timeout if timeout is not None else self.default_timeout
        conn: dbapi.Connection
        try:
            with self.create_connection() as conn:
                cursor = conn.cursor()
                try:
                    # Execute query
                    cursor.execute(query)

                    # Get results
                    results = cursor.fetchall()

                    return [
                        dict(zip(row.column_names, row.column_values))
                        for row in results
                    ]

                except Exception as e:
                    # Handle SAP Data Sphere specific errors
                    raise InvalidGeneratedCode(
                        f"SAP Data Sphere error: {str(e)}",
                        code=query,
                        exception=None,
                        traceback_str="",
                    )
                finally:
                    cursor.close()

        except Exception as e:
            raise InvalidGeneratedCode(
                f"Query execution failed: {str(e)}",
                code=query,
                exception=e,
                traceback_str=traceback.format_exc(),
            )

    def get_tables(self, timeout: int | None = None) -> list[str]:
        """Fetch list of tables from SAP Data Sphere schema"""
        timeout = timeout if timeout is not None else self.default_timeout

        conn: dbapi.Connection
        try:
            with self.create_connection() as conn:
                cursor = conn.cursor()
                try:
                    # Get all tables and views in the schema
                    cursor.execute(
                        f"""
                        SELECT TABLE_NAME 
                        FROM SYS.TABLES 
                        WHERE SCHEMA_NAME = '{self._credentials.db_schema}'
                        ORDER BY TABLE_NAME
                        """
                    )
                    tables = [row[0] for row in cursor.fetchall()]

                    # Get all views
                    cursor.execute(
                        f"""
                        SELECT VIEW_NAME 
                        FROM SYS.VIEWS 
                        WHERE SCHEMA_NAME = '{self._credentials.db_schema}'
                        ORDER BY VIEW_NAME
                        """
                    )
                    views = [row[0] for row in cursor.fetchall()]

                    all_objects = tables + views

                    # Log detailed results
                    logger.info(
                        f"Total objects found in schema {self._credentials.db_schema}: {len(all_objects)}"
                    )
                    logger.info(f"Tables: {len(tables)}, Views: {len(views)}")

                    return all_objects

                finally:
                    cursor.close()

        except Exception as e:
            logger.error(f"Failed to fetch tables from SAP Data Sphere: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {str(e)}")
            return []

    @functools.lru_cache(maxsize=8)
    async def get_data(
        self,
        *table_names: str,
        analyst_db: AnalystDB,
        sample_size: int = 5000,
        timeout: int | None = None,
    ) -> list[str]:
        """Load selected tables from SAP Data Sphere as DataFrames

        Args:
        - table_names: List of table names to fetch
        - sample_size: Number of rows to sample from each table
        - timeout: Query timeout in seconds

        Returns:
        - List of registered dataset names
        """
        timeout = timeout if timeout is not None else self.default_timeout
        dataframes = []

        try:
            with self.create_connection() as conn:
                cursor = conn.cursor()

                for table in table_names:
                    try:
                        qualified_table = f'"{self._credentials.db_schema}"."{table}"'
                        logger.info(f"Fetching data from table: {qualified_table}")

                        # Execute query to get data with limit
                        cursor.execute(
                            f"""
                            SELECT * FROM {qualified_table}
                            LIMIT {sample_size}
                            """
                        )

                        # Get column names
                        columns = [desc[0] for desc in cursor.description]
                        data = cursor.fetchall()

                        # Convert to pandas DataFrame
                        pandas_df = pd.DataFrame(data=data, columns=columns, dtype=str)

                        # Convert to polars DataFrame
                        df = pl.DataFrame(
                            data=pandas_df, schema={col: pl.String for col in columns}
                        )

                        logger.info(
                            f"Successfully loaded table {table}: {len(df)} rows, {len(df.columns)} columns"
                        )
                        dataframes.append(AnalystDataset(name=table, data=df))

                    except Exception as e:
                        logger.error(f"Error loading table {table}: {str(e)}")
                        logger.error(f"Error type: {type(e)}")
                        logger.error(f"Error details: {str(e)}")
                        continue

                # Register datasets
                names = []
                for dataframe in dataframes:
                    await analyst_db.register_dataset(
                        dataframe, DataSourceType.DATABASE
                    )
                    names.append(dataframe.name)
                return names

        except Exception as e:
            logger.error(f"Error fetching SAP Data Sphere data: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {str(e)}")
            return []

    def get_system_prompt(self) -> ChatCompletionSystemMessageParam:
        return ChatCompletionSystemMessageParam(
            role="system",
            content=SYSTEM_PROMPT_SAP_DATASPHERE.format(
                schema=self._credentials.db_schema,
            ),
        )


def get_database_operator(app_infra: AppInfra) -> DatabaseOperator[Any]:
    if app_infra.database == "bigquery":
        credentials: (
            GoogleCredentialsBQ
            | SnowflakeCredentials
            | SAPDatasphereCredentials
            | DataBricksCredentials # ADDED FOR DATABRICKS
            | NoDatabaseCredentials
        )
        try:
            credentials = GoogleCredentialsBQ()
            if credentials.service_account_key and credentials.db_schema:
                return BigQueryOperator(credentials)
        except (ValidationError, ValueError):
            logger.warning(
                "BigQuery credentials not properly configured, falling back to no database"
            )
        return NoDatabaseOperator(NoDatabaseCredentials())
    elif app_infra.database == "snowflake":
        try:
            credentials = SnowflakeCredentials()
            if credentials.is_configured():
                return SnowflakeOperator(credentials)
        except (ValidationError, ValueError):
            logger.warning(
                "Snowflake credentials not properly configured, falling back to no database"
            )
        return NoDatabaseOperator(NoDatabaseCredentials())
    elif app_infra.database == "sap":
        try:
            credentials = SAPDatasphereCredentials()
            if credentials.is_configured():
                return SAPDatasphereOperator(credentials)
        except (ValidationError, ValueError):
            logger.warning(
                "SAP credentials not properly configured, falling back to no database"
            )
        return NoDatabaseOperator(NoDatabaseCredentials())
    elif app_infra.database == "databricks": # ADDED FOR DATABRICKS
        try:
            credentials = DataBricksCredentials()
            if credentials.is_configured():
                return DataBricksSqlOperator(credentials)
        except (ValidationError, ValueError):
            logger.warning(
                "DataBricks credentials not properly configured, falling back to no database"
            )
        return NoDatabaseOperator(NoDatabaseCredentials())
    else:
        return NoDatabaseOperator(NoDatabaseCredentials())


def load_app_infra() -> AppInfra:
    directories = [".", "frontend", "app_backend"]
    error = None
    for directory in directories:
        path = Path(directory).joinpath("app_infra.json")
        try:
            with open(path) as infra_selection:
                app_json = json.load(infra_selection)
                return AppInfra(**app_json)
        except (FileNotFoundError, ValidationError) as e:
            error = e
    raise ValueError(
        "Failed to read app_infra.json.\n"
        "If running locally, verify you have selected the correct "
        "stack and that it is active using `pulumi stack output`.\n"
        f"Ensure file is created by running `pulumi up`: {str(error)}"
    ) from error


def get_external_database() -> DatabaseOperator[Any]:
    return get_database_operator(load_app_infra())


#### ADDED FOR DATABRICKS ####
class DataBricksSqlOperator(DatabaseOperator[DataBricksCredentialArgs]): #DatabaseOperator["DataBricksCredentials"]):
    """DataBricks SQL server operator using databricks-sql-connector for pure Python implementation"""

    def __init__(
        self,
        credentials: DataBricksCredentials,
        default_timeout: int = _DEFAULT_DB_QUERY_TIMEOUT,
    ):
        if not credentials.is_configured():
            raise ValueError("DataBricks credentials not properly configured")
        self._credentials = credentials
        self.default_timeout = default_timeout

    @contextmanager
    def create_connection(self) -> Generator[sql.Connection, None, None]:
        """Create a connection to DataBricks SQL server"""
        if not self._credentials.is_configured():
            raise ValueError("Databricks credentials not properly configured")

        def credential_provider(config):
            return oauth_service_principal(config)

        config = Config(
            host=self._credentials.server_hostname,
            client_id=self._credentials.client_id,
            client_secret=self._credentials.client_secret,
            tenant_id=self._credentials.tenant_id,
        )
      
        try:
            connection = sql.connect(
                server_hostname=self._credentials.server_hostname,
                http_path=self._credentials.http_path,
                credentials_provider=lambda: credential_provider(config),
                config=config
            )
            yield connection
        except Exception as e:
            print(f"Failed to connect to Databricks: {e}")
            raise
        finally:
            if 'connection' in locals() and connection:
                connection.close()

    
    def execute_query(
        self, query: str, timeout: int | None = None
    ) -> Union[List[tuple], List[Dict[str, Any]]]: # UPDATE IF NEEDED
        """Execute a Databricks SQL query with a timeout and metadata capture.
    
        Args:
            query: SQL query to execute
            timeout: Query timeout in seconds (not directly supported, for future use)
    
        Returns:
            List of results as either tuples or dictionaries.
        """
        timeout = timeout if timeout is not None else self.default_timeout
 
        try:
            with self.create_connection() as conn:
                with conn.cursor() as cursor:
                    try:
                        # Set query timeout at cursor level
                        cursor.execute(
                            f"SET STATEMENT_TIMEOUT = {timeout}"
                        )
                        
                        # Execute query
                        cursor.execute(query)
    
                        # Get results
                        results = cursor.fetchall()
                        # The databricks-sql-connector returns results as Row objects.
                        # Convert them to dictionaries:
                        # # Get column names from cursor description
                        # columns = [col[0] for col in cursor.description]
                        # Convert each Row object to a dictionary
                        results = [row.asDict() for row in results]
    
                        return results
    
                    except Exception as e:
                        # Handle Databricks-specific errors
                        raise InvalidGeneratedCode(
                            f"Databricks error: {str(e)}",
                            code=query,
                            exception=None,
                            traceback_str="",
                        )
                        
        except Exception as e:
            raise InvalidGeneratedCode(
                f"Query execution failed: {str(e)}",
                code=query,
                exception=e,
                traceback_str="" #traceback.format_exc(),
            )


    def get_tables(self, timeout: Optional[int] = None) -> List[str]:
        """Fetch list of tables and views from a Databricks schema."""
        timeout = timeout if timeout is not None else self.default_timeout
        
        try:
            with self.create_connection() as conn:
                with conn.cursor() as cursor:
                    # Log current session info
                    logger.info("Checking current session settings...")
                    # Set query timeout at cursor level
                    cursor.execute(
                        f"SET STATEMENT_TIMEOUT = {timeout}"
                    )
                    
                    # Databricks does not have a simple equivalent to Snowflake's CURRENT_DATABASE/SCHEMA/ROLE/WAREHOUSE
                    # Check if the schema exists in Databricks
                    schema_catalog = self._credentials.catalog
                    schema_name = self._credentials.db_schema
    
                    #Get all tables and views from the specified schema
                    cursor.execute(f"SHOW TABLES IN {schema_catalog}.{schema_name}")
                    
                    results = cursor.fetchall()
                    
                    # Use asDict() to get a dictionary, then extract the 'table_name' key
                    tables = [row.asDict()['tableName'] for row in results]

                    # # Log detailed results
                    # logger.info(f"Total objects found: {len(results)}")
                    # for row in results:
                    #     row_dict = row.asDict()
                    #     logger.info(f"Found {row_dict['table_type']}: {row_dict['table_name']}")
    
                    # Note: The SHOW GRANTS syntax is specific and may not work as a direct SQL query.
                    # This part is omitted for correctness.
                    
                    return tables
    
        except Exception as e:
            logger.error(f"Failed to fetch tables: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            return []
    

    @functools.lru_cache(maxsize=8)
    async def get_data(
        self,
        *table_names: str,
        analyst_db: AnalystDB,
        sample_size: int = 5000,
        timeout: Optional[int] = None,
    ) -> List[str]:
        """Load selected tables from Databricks as Polars DataFrames
    
        Args:
        - table_names: List of table names to fetch
        - analyst_db: The AnalystDB object to register datasets with
        - sample_size: Number of rows to sample from each table
    
        Returns:
        - List of table names that were successfully loaded
        """
        timeout = timeout if timeout is not None else self.default_timeout

        dataframes = []
        try:
            with self.create_connection() as conn:
                for table in table_names:
                    try:
                        # Construct the qualified table name for Databricks
                        qualified_table = f"{self._credentials.catalog}.{self._credentials.db_schema}.{table}"
                        logger.info(f"Fetching data from table: {qualified_table}")
                        with conn.cursor() as cursor:
                            # Set query timeout at cursor level
                            cursor.execute(
                                f"SET STATEMENT_TIMEOUT = {timeout}"
                            )
    
                            # Databricks uses TABLESAMPLE instead of SAMPLE
                            # Also, the query timeout is handled at the SQL endpoint level
                            # The `timeout` parameter is not directly used in the query.
                            query = f"""
                            SELECT * FROM {qualified_table}
                            TABLESAMPLE ({sample_size} ROWS)
                            """
    
                            cursor.execute(query)
                            
                            # Get column names
                            columns = [desc[0] for desc in cursor.description]
                            
                            # Fetch all rows. This returns a list of Row objects.
                            rows = cursor.fetchall()
                            
                            # Convert Row objects to a list of dictionaries for easy DataFrame creation
                            data = [row.asDict() for row in rows]
                            
                            # Create a pandas DataFrame from the data
                            pandas_df = pd.DataFrame(data=data, columns=columns)
                            
                            # Convert to Polars DataFrame
                            df = pl.DataFrame(
                                data=pandas_df, schema={col: pl.String for col in columns}
                            )
    
                            logger.info(
                                f"Successfully loaded table {table}: {len(df)} rows, {len(df.columns)} columns"
                            )
                            dataframes.append(AnalystDataset(name=table, data=df))
    
                    except Exception as e:
                        logger.error(f"Error loading table {table}: {str(e)}")
                        logger.error(f"Error type: {type(e)}")
                        logger.error(f"Error details: {traceback.format_exc()}")
                        continue
    
                names = []
                for dataframe in dataframes:
                    await analyst_db.register_dataset(
                        dataframe, DataSourceType.DATABASE
                    )
                    names.append(dataframe.name)
                return names
    
        except Exception as e:
            logger.error(f"Error fetching DataBricks data: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {str(e)}")
            return []
    
    def get_system_prompt(self) -> Any:
        """Get the system prompt for DataBricks Server T-SQL code generation"""
        # from openai.types.chat import ChatCompletionSystemMessageParam

        return ChatCompletionSystemMessageParam(
            role="system",
            content=SYSTEM_PROMPT_DATABRICKS.format(
                catalog=self._credentials.catalog,
                schema=self._credentials.db_schema,
            ),
        )