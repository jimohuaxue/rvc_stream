#!/usr/bin/env python3
"""
RVC Server with Admin Panel
Starts the WebSocket server with integrated admin panel
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Also add parent directory foradmin module
parent_dir = Path(__file__).parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


def main():
    """Main entry point"""
    import argparse
    import logging
    from src.rvc_server import RVCStreamingServer, DEFAULT_PORT

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="RVC WebSocket Streaming Server with Admin Panel"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to bind to"
    )
    parser.add_argument("--max-cache", type=int, default=3, help="Max cached models")
    args = parser.parse_args()

    # Create server
    server = RVCStreamingServer(host=args.host, port=args.port)
    server.model_manager.max_cache = args.max_cache

    # Setup admin panel
    try:
        from src.admin import init_admin_panel, setup_log_streaming

        # Initialize admin panel with server instance
        init_admin_panel(server)

        # Setup log streaming
        setup_log_streaming(logger)

        # Import and mount admin routes
        from src.admin import admin_router

        server.app.include_router(
            admin_router
        )  # Prefix already set in router definition

        logger.info("Admin panel mounted at /admin")

        # Apply persisted RVC params to processor so saved config survives restarts
        try:
            from src.admin.config_manager import get_config_manager
            rvc_cfg = get_config_manager().get_rvc_config()
            server.processor.update_rvc_params(rvc_cfg.model_dump())
            logger.info("RVC params loaded from config")
        except Exception as e:
            logger.warning(f"Could not load RVC params from config: {e}")

    except ImportError as e:
        logger.warning(f"Admin panel not available: {e}")
    except Exception as e:
        logger.error(f"Failed to setup admin panel: {e}")

    # Run server
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()
