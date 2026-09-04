import smtplib


def test_signup_otp_and_signin_flow(client, monkeypatch):
    class DummySMTP:
        def __init__(self, *args, **kwargs):
            self.sent = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def starttls(self):
            return None

        def login(self, *args, **kwargs):
            return None

        def send_message(self, message):
            self.sent.append(message)

    monkeypatch.setenv("SMTP_USER", "noreply@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "supersecret")
    monkeypatch.setattr(smtplib, "SMTP", DummySMTP)

    signup_response = client.get(
        "/signup?user=alice&email=alice@example.com&password=StrongPass123"
    )
    assert signup_response.status_code == 200
    assert b"otp" in signup_response.data.lower()

    with client.session_transaction() as session:
        pending_signup = session["pending_signup"]
        otp_code = pending_signup["otp"]

    otp_response = client.post("/otp", data={"message": str(otp_code)})
    assert otp_response.status_code == 200
    assert b"Account created successfully" in otp_response.data

    login_response = client.post(
        "/signin",
        data={"user": "alice", "password": "StrongPass123"},
        follow_redirects=False,
    )
    assert login_response.status_code == 302
    assert login_response.headers["Location"].endswith("/home")


def test_invalid_signin_and_logout(client):
    invalid_response = client.post(
        "/signin",
        data={"user": "unknownuser", "password": "WrongPass123"},
        follow_redirects=True,
    )
    assert invalid_response.status_code == 200
    assert b"Invalid username/email or password" in invalid_response.data

    client.post(
        "/signin",
        data={"user": "admin", "password": "admin123"},
        follow_redirects=False,
    )

    logout_response = client.get("/logout", follow_redirects=False)
    assert logout_response.status_code == 302
    assert logout_response.headers["Location"].endswith("/login")
