# DataBricks Integration Readme for Talk to My Data Agent

## Overview

This readme documents the modifications made to add DataBricks SQL Server support to the Talk to My Data Agent application. This is an early stage implementation that can allow TTMData to connect to a DataBricks SQL Server and perform data analysis as per TTMData functionality. Developers should further modify this modified template according to their business requirements (e.g. security, integration, and business context).

In addition to the DataBricks integration, this repo also includes a patch for TTMData version 3.11 to improve performance. The patch is not required to run the DataBricks connection but recommended for improved performance to TTMData (versions 3.11 and earlier).

## TTMData Version

This integration was developed on top of TTMData version 3.11 (2025-09-22).

## Modifications

The following files have been modified as described:

### utils/database_helpers.py
- Added additional import from typing
- Added import from databricks sdk
- Added import to other related objects in the TTMData DataBricks code (i.e. DataBricksCredentials, SYSTEM_PROMPT_DATABRICKS)
- Added DataBricksCredentialsArg dataclass
- Added DataBricks SQL Server operator (DataBricksSqlOperator) with the following functions implemented for DataBricks:
    + init
    + create_connection
    + execute_query
    + get_tables
    + get_data
    + get_system_prompt
        + Reference SYSTEM_PROMPT_DATABRICKS from utils/prompts.py
- Modified get_database_operator to accept and validate DataBricksCredentials

### utils/credentials.py
- Added DataBricksCredentials class with the following attributes:
    + server_hostname
    + client_id
    + client_secret
    + tenant_id
    + http_path
    + catalog
    + db_schema
    + is_configured

### utils/prompts.py
- Added SYSTEM_PROMPT_DATABRICKS

### utils/schema.py
- Added "databricks" to DatabaseConnectionType

### infra/settings_database.py
- Updated DATABASE_CONNECTION_TYPE to use "databricks"

### infra/components/dr_credential.py
- Added import to other related objects in the TTMData DataBricks code (i.e. DataBricksCredentials, SYSTEM_PROMPT_DATABRICKS)
- Added DataBricks credential handling in get_credential_runtime_parameter_values function
- Modified get_database_credentials to accept and validate DataBricksCredentials

### Requirements.txt
The following libraries were added:
- databricks-sdk
- databricks-sql-connector

The following requirements.txt were updated accordingly:
- requirements.txt (in main folder)
- app_backend/requirements.txt
- frontend/requirements.txt

### .env
- Added the following DataBricks variables:
    + DATABRICKS_SERVER_HOSTNAME
    + DATABRICKS_CLIENT_ID
    + DATABRICKS_CLIENT_SECRET
    + DATABRICKS_TENANT_ID
    + DATABRICKS_HTTP_PATH
    + DATABRICKS_CATALOG
    + DATABRICKS_DB_SCHEMA

## Performance patch

This repo also includes a patch (performance_patch/Improvements_to_sync_debug_logging.patch) for TTMData version 3.11 and earlier to improve app performance. The patch improves the implementation of Fast API in the core application.

To apply the patch:

1. Move the patch file to the main folder

2. Run the following in a terminal:

```
git apply Improvements_to_sync_debug_logging.patch
```

## Considerations
