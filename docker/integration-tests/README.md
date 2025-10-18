[38;2;127;132;156m   1[0m [38;2;205;214;244m# Integration Tests Docker Compose Environment[0m
[38;2;127;132;156m   2[0m 
[38;2;127;132;156m   3[0m [38;2;205;214;244mThis Docker Compose configuration provides a multi-component testing environment for the Workspace Qdrant MCP server and Rust daemon.[0m
[38;2;127;132;156m   4[0m 
[38;2;127;132;156m   5[0m [38;2;205;214;244m## Architecture[0m
[38;2;127;132;156m   6[0m 
[38;2;127;132;156m   7[0m [38;2;205;214;244mThe environment consists of the following services:[0m
[38;2;127;132;156m   8[0m 
[38;2;127;132;156m   9[0m [38;2;205;214;244m1. **Qdrant** - Vector database for storing and querying embeddings[0m
[38;2;127;132;156m  10[0m [38;2;205;214;244m2. **Daemon** - Rust daemon for file watching and automatic ingestion[0m
[38;2;127;132;156m  11[0m [38;2;205;214;244m3. **MCP Server** - Python FastMCP server exposing tools for Claude[0m
[38;2;127;132;156m  12[0m [38;2;205;214;244m4. **Test Runner** - Pytest test execution environment (optional profile)[0m
[38;2;127;132;156m  13[0m [38;2;205;214;244m5. **Performance Monitor** - Performance metrics collection (optional profile)[0m
[38;2;127;132;156m  14[0m 
[38;2;127;132;156m  15[0m [38;2;205;214;244m## Quick Start[0m
[38;2;127;132;156m  16[0m 
[38;2;127;132;156m  17[0m [38;2;205;214;244m### Start All Core Services[0m
[38;2;127;132;156m  18[0m 
[38;2;127;132;156m  19[0m [38;2;205;214;244m```bash[0m
[38;2;127;132;156m  20[0m [38;2;205;214;244mcd docker/integration-tests[0m
[38;2;127;132;156m  21[0m [38;2;205;214;244mdocker-compose up -d[0m
[38;2;127;132;156m  22[0m [38;2;205;214;244m```[0m
[38;2;127;132;156m  23[0m 
[38;2;127;132;156m  24[0m [38;2;205;214;244mThis starts:[0m
[38;2;127;132;156m  25[0m [38;2;205;214;244m- Qdrant (ports 6333, 6334)[0m
[38;2;127;132;156m  26[0m [38;2;205;214;244m- Rust daemon[0m
[38;2;127;132;156m  27[0m [38;2;205;214;244m- MCP server (port 8000)[0m
[38;2;127;132;156m  28[0m 
[38;2;127;132;156m  29[0m [38;2;205;214;244m### Verify Services[0m
[38;2;127;132;156m  30[0m 
[38;2;127;132;156m  31[0m [38;2;205;214;244m```bash[0m
[38;2;127;132;156m  32[0m [38;2;205;214;244m# Check all services are healthy[0m
[38;2;127;132;156m  33[0m [38;2;205;214;244mdocker-compose ps[0m
[38;2;127;132;156m  34[0m 
[38;2;127;132;156m  35[0m [38;2;205;214;244m# View logs[0m
[38;2;127;132;156m  36[0m [38;2;205;214;244mdocker-compose logs -f[0m
[38;2;127;132;156m  37[0m 
[38;2;127;132;156m  38[0m [38;2;205;214;244m# View individual service logs[0m
[38;2;127;132;156m  39[0m [38;2;205;214;244mdocker-compose logs -f mcp-server[0m
[38;2;127;132;156m  40[0m [38;2;205;214;244mdocker-compose logs -f daemon[0m
[38;2;127;132;156m  41[0m [38;2;205;214;244mdocker-compose logs -f qdrant[0m
[38;2;127;132;156m  42[0m [38;2;205;214;244m```[0m
[38;2;127;132;156m  43[0m 
[38;2;127;132;156m  44[0m [38;2;205;214;244m## Service Details[0m
[38;2;127;132;156m  45[0m 
[38;2;127;132;156m  46[0m [38;2;205;214;244m### Qdrant[0m
[38;2;127;132;156m  47[0m [38;2;205;214;244m- **Ports**: 6333 (HTTP), 6334 (gRPC)[0m
[38;2;127;132;156m  48[0m [38;2;205;214;244m- **Health Check**: `/healthz` endpoint[0m
[38;2;127;132;156m  49[0m [38;2;205;214;244m- **Data**: Persisted in `qdrant_storage` volume[0m
[38;2;127;132;156m  50[0m 
[38;2;127;132;156m  51[0m [38;2;205;214;244m### Daemon[0m
[38;2;127;132;156m  52[0m [38;2;205;214;244m- **Binary**: `memexd`[0m
[38;2;127;132;156m  53[0m [38;2;205;214;244m- **Purpose**: Watches SQLite for configuration changes, ingests files[0m
[38;2;127;132;156m  54[0m [38;2;205;214;244m- **Dependencies**: Qdrant must be healthy[0m
[38;2;127;132;156m  55[0m [38;2;205;214;244m- **Data**: State stored in `daemon_data` volume[0m
[38;2;127;132;156m  56[0m 
[38;2;127;132;156m  57[0m [38;2;205;214;244m### MCP Server[0m
[38;2;127;132;156m  58[0m [38;2;205;214;244m- **Port**: 8000[0m
[38;2;127;132;156m  59[0m [38;2;205;214;244m- **Health**: `/health` endpoint[0m
[38;2;127;132;156m  60[0m [38;2;205;214;244m- **Dependencies**: Both Qdrant and Daemon must be healthy[0m
[38;2;127;132;156m  61[0m 
[38;2;127;132;156m  62[0m [38;2;205;214;244m## Volumes[0m
[38;2;127;132;156m  63[0m 
[38;2;127;132;156m  64[0m [38;2;205;214;244mAll data is persisted in Docker volumes:[0m
[38;2;127;132;156m  65[0m [38;2;205;214;244m- `qdrant_storage`, `daemon_data`, `mcp_data`, etc.[0m
[38;2;127;132;156m  66[0m 
[38;2;127;132;156m  67[0m [38;2;205;214;244m## Cleanup[0m
[38;2;127;132;156m  68[0m 
[38;2;127;132;156m  69[0m [38;2;205;214;244m```bash[0m
[38;2;127;132;156m  70[0m [38;2;205;214;244m# Stop services[0m
[38;2;127;132;156m  71[0m [38;2;205;214;244mdocker-compose down[0m
[38;2;127;132;156m  72[0m 
[38;2;127;132;156m  73[0m [38;2;205;214;244m# Stop and remove volumes[0m
[38;2;127;132;156m  74[0m [38;2;205;214;244mdocker-compose down -v[0m
[38;2;127;132;156m  75[0m [38;2;205;214;244m```[0m
