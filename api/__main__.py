"""Run the dashboard with settings from the environment: `python -m api`.

Equivalent to `uvicorn api.main:app --host $HOST --port $PORT --log-level $LOG_LEVEL`,
so the same command works locally, in Docker and on hosts that set $PORT.
"""
import uvicorn

from api.config import settings

if __name__ == "__main__":
    uvicorn.run("api.main:app", host=settings.host, port=settings.port, log_level=settings.log_level,
                # Behind a host's reverse proxy: trust X-Forwarded-* so URLs and client IPs are right
                proxy_headers=True, forwarded_allow_ips="*")
