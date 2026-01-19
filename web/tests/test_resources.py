
"""
Resource Management Tests (KB & Documents)
"""

import pytest
from pathlib import Path

class TestResourceManagement:

    def test_kb_lifecycle_full(self, test_client, default_embedder_name):
        """Test full KB lifecycle: Create -> Update -> Delete."""
        # 1. Create
        kb_data = {
            "name": "Lifecycle KB",
            "description": "Initial Description",
            "embedder_name": default_embedder_name,
        }
        res_create = test_client.post("/api/kb", json=kb_data)
        assert res_create.status_code == 200
        kb = res_create.json()
        kb_id = kb["id"]
        assert kb["name"] == "Lifecycle KB"
        assert kb["description"] == "Initial Description"

        # 2. Update Basic Info
        res_update = test_client.put(f"/api/kb/{kb_id}", json={
            "description": "Updated Description",
        })
        assert res_update.status_code == 200
        updated_kb = res_update.json()
        assert updated_kb["description"] == "Updated Description"

        # 3. List KBs to verify presence
        res_list = test_client.get("/api/kb")
        kbs = res_list.json()
        assert any(k["id"] == kb_id for k in kbs)

        # 4. Delete
        res_del = test_client.delete(f"/api/kb/{kb_id}")
        assert res_del.status_code == 200

        # 5. Verify Gone
        res_get = test_client.get(f"/api/kb/{kb_id}")
        assert res_get.status_code == 404

    def test_document_listing(self, test_client, test_kb_id, sample_document):
        """Test listing documents in a KB."""
        # 1. Upload a document first
        with open(sample_document, 'rb') as f:
            res_upload = test_client.post(
                "/api/upload",
                data={"kb_id": test_kb_id},
                files={"files": (sample_document.name, f, "text/markdown")}
            )
        assert res_upload.status_code == 200
        
        # 2. List documents
        # Note: Endpoint is /api/upload/documents/{kb_id}
        res_list = test_client.get(f"/api/upload/documents/{test_kb_id}")
        assert res_list.status_code == 200
        docs = res_list.json()
        
        assert isinstance(docs, list)
        assert len(docs) > 0
        assert any(d["filename"] == sample_document.name for d in docs)
        
        # Verify fields
        doc = docs[0]
        assert "id" in doc
        assert "chunk_count" in doc

