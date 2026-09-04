from io import BytesIO


def test_bulk_upload_valid_csv(client):
    csv_data = b"review_text\nThe service was excellent and the staff were friendly.\nRoom was dirty and the wait time was terrible.\n"
    response = client.post(
        "/bulk_upload",
        data={"file": (BytesIO(csv_data), "reviews.csv")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Bulk Analysis Results" in response.data or b"total_count" in response.data or b"total_count" in response.data


def test_bulk_upload_invalid_file(client):
    response = client.post(
        "/bulk_upload",
        data={"file": (BytesIO(b"not a valid csv file"), "notes.txt")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Unsupported file format" in response.data or b"Please select a valid CSV or Excel file" in response.data
