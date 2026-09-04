def test_admin_routes_require_admin_login(client):
    protected_routes = ["/admin", "/admin/analytics-export", "/admin/export-analytics-csv", "/admin/raw-data"]
    for route in protected_routes:
        response = client.get(route, follow_redirects=False)
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/admin/login")


def test_admin_login_dashboard_and_edit_delete_flow(client):
    login_response = client.post(
        "/admin/login",
        data={"username": "admin", "password": "admin123"},
        follow_redirects=False,
    )
    assert login_response.status_code == 302
    assert login_response.headers["Location"].endswith("/admin")

    dashboard_response = client.get("/admin")
    assert dashboard_response.status_code == 200
    assert b"dashboard" in dashboard_response.data.lower()

    client.post(
        "/signin",
        data={"user": "admin", "password": "admin123"},
        follow_redirects=False,
    )

    user_insert = client.post(
        "/signup?user=beta&email=beta@example.com&password=StrongPass123",
        follow_redirects=False,
    )
    assert user_insert.status_code == 200
    with client.session_transaction() as session:
        otp = session["pending_signup"]["otp"]

    client.post("/otp", data={"message": str(otp)})

    with client.application.app_context():
        from app import connect_db

        conn = connect_db()
        user = conn.execute("SELECT id FROM info WHERE user = ?", ("beta",)).fetchone()
        conn.close()
        assert user is not None

    edit_response = client.post(
        f"/admin/edit/{user[0]}",
        data={"user": "beta-updated", "name": "Beta User", "email": "beta@example.com", "mobile": "9876543210", "role": "user"},
        follow_redirects=False,
    )
    assert edit_response.status_code == 302
    assert edit_response.headers["Location"].endswith("/admin")

    delete_response = client.post(f"/admin/delete/{user[0]}", follow_redirects=False)
    assert delete_response.status_code == 302
    assert delete_response.headers["Location"].endswith("/admin")
