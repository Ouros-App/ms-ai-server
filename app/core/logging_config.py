import logging


def configure_logging() -> None:
    """Configura logs legíveis e sem conteúdo de mensagens ou credenciais."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        force=True,
    )
