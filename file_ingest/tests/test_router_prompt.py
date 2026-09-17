"""
test_router_prompt.py —— has_file_intent + build_file_list_prompt 集成测试
"""
from __future__ import annotations

import json
import pytest


class TestHasFileIntent:
    def test_attachments_field(self):
        from file_ingest import has_file_intent
        assert has_file_intent({"attachments": ["/tmp/a.pdf"]}) is True
        assert has_file_intent({"attachments": []}) is False

    def test_file_id_field(self):
        from file_ingest import has_file_intent
        assert has_file_intent({"file_id": "abc-123"}) is True

    def test_ext_in_text(self):
        from file_ingest import has_file_intent
        assert has_file_intent("看下 report.pdf") is True
        assert has_file_intent("分析 data.xlsx") is True
        assert has_file_intent("读 src/main.py") is True

    def test_explicit_action(self):
        from file_ingest import has_file_intent
        assert has_file_intent("帮我上传一个文档解析一下") is True

    def test_negative(self):
        from file_ingest import has_file_intent
        assert has_file_intent("今天天气怎么样") is False
        assert has_file_intent("帮我写首诗") is False
        assert has_file_intent({"content": "普通对话"}) is False

    def test_file_id_pattern_in_text(self):
        from file_ingest import has_file_intent
        # file_id 引用形式
        assert has_file_intent('请基于 file_id="abc-123-456" 回答') is True


class TestBuildFileListPrompt:
    def test_empty_returns_empty(self, configured):
        from file_ingest import build_file_list_prompt
        assert build_file_list_prompt(configured, []) == ""

    def test_includes_active_files(self, configured):
        from file_ingest import build_file_list_prompt, register_file
        register_file("fid-1", "report.pdf", "pdf", "季度报告摘要")
        register_file("fid-2", "code.py", "py", "示例代码")

        prompt = build_file_list_prompt(configured, ["fid-1", "fid-2"])
        assert "report.pdf" in prompt
        assert "code.py" in prompt
        assert "季度报告摘要" in prompt
        assert prompt.startswith("你有以下文件可用")

    def test_unknown_file_id(self, configured):
        from file_ingest import build_file_list_prompt
        prompt = build_file_list_prompt(configured, ["nonexistent"])
        assert "nonexistent" in prompt  # 降级展示
