import pytest
from httpx import AsyncClient, ASGITransport


from clarion.app import app  # adjust import to match your actual path

@pytest.mark.asyncio
async def test_predict_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/predict", json={
  "Pclass": 3,
  "Sex": "female",
  "Age": 22,
  "SibSp": 4,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S"
})
    assert response.status_code == 200
    assert "Survived" in response.json()