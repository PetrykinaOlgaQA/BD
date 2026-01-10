from .base import BaseRepository
from sqlalchemy import text

class UserRepository(BaseRepository):
    def get_by_email(self, email: str):
        return self.conn.execute(
            text("SELECT id, username, email, password FROM Users WHERE email = :email"),
            {"email": email}
        ).fetchone()

    def create(self, username: str, email: str, password: str):
        self.conn.execute(
            text("INSERT INTO Users (username, email, password) VALUES (:username, :email, :password)"),
            {"username": username, "email": email, "password": password}
        )

    def exists_by_email_or_username(self, email: str, username: str):
        return self.conn.execute(
            text("SELECT 1 FROM Users WHERE email = :email OR username = :username"),
            {"email": email, "username": username}
        ).fetchone() is not None

    def get_all(self):
        return self.conn.execute(text("SELECT Username, Email FROM Users")).fetchall()