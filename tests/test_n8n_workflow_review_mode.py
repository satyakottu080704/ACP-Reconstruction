import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "automation" / "n8n" / "acorn_plans_renderer.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _node(workflow, name):
    for node in workflow.get("nodes", []):
        if node.get("name") == name:
            return node
    raise AssertionError(f"missing node: {name}")


def test_renderer_workflow_is_review_only_and_uses_env_token():
    workflow = _load(RENDERER)

    assert workflow.get("active") is False
    assert "review" in workflow.get("name", "").lower()

    render_node = _node(workflow, "Render (POST 8765)")
    headers = render_node["parameters"]["headerParameters"]["parameters"]
    token = next(h["value"] for h in headers if h["name"] == "X-Auth-Token")
    assert token == "={{ $env.RENDER_SERVICE_TOKEN }}"

    upload_node = _node(workflow, "Upload to SharePoint Manual_Review")
    upload_url = upload_node["parameters"]["url"]
    assert "Manual_Review" in upload_url
    assert "Generated_Plans" not in upload_url
    assert "General/AI Automation/Completed" not in upload_url

    node_names = {node.get("name") for node in workflow.get("nodes", [])}
    assert "Move email to Completed" in node_names
    assert "Upload (Completed)" not in node_names
    assert "Clean? (no review needed)" not in node_names
