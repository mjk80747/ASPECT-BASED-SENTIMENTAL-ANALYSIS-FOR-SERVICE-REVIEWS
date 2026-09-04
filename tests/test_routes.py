def test_root_and_about_routes(client):
    index_response = client.get("/")
    assert index_response.status_code == 200
    assert b"service" in index_response.data.lower()

    about_response = client.get("/about")
    assert about_response.status_code == 200
    assert b"about" in about_response.data.lower()


def test_analytics_route_loads(client):
    response = client.get("/analytics")
    assert response.status_code == 200
    assert b"analytics" in response.data.lower()


def test_predict_valid_and_empty_input(client):
    valid_response = client.post(
        "/predict",
        data={"message": "The food was amazing and the staff were friendly."},
        follow_redirects=True,
    )
    assert valid_response.status_code == 200
    assert b"Review" in valid_response.data or b"pred_output" in valid_response.data

    empty_response = client.post("/predict", data={"message": "   "}, follow_redirects=True)
    assert empty_response.status_code == 400
    assert b"Please enter a message to analyze" in empty_response.data
