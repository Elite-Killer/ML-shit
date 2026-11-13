import json

from dm_ahr_vit.config import load_hair_screen_config
from dm_ahr_vit.paths import REPO_ROOT


def test_shipped_config_is_hash_consistent():
    assert load_hair_screen_config()["config_hash"]


def test_shipped_config_equals_the_committed_frozen_run_config():
    frozen = json.loads((REPO_ROOT / "artifacts/logs/hair_frozen_config.json").read_text())
    assert load_hair_screen_config() == frozen["screen"]
