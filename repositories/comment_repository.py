from .base import BaseRepository
from sqlalchemy import text

class CommentRepository(BaseRepository):
    def get_by_post_id(self, post_id: int):
        return self.conn.execute(text("""
            SELECT c.id, c.content, c.created_date, u.username AS author, c.user_id
            FROM Comments c
            JOIN Users u ON c.user_id = u.id
            WHERE c.post_id = :id
            ORDER BY c.created_date ASC
        """), {"id": post_id}).fetchall()

    def create(self, post_id: int, user_id: int, content: str):
        self.conn.execute(
            text("INSERT INTO Comments (post_id, user_id, content) VALUES (:post_id, :user_id, :content)"),
            {"post_id": post_id, "user_id": user_id, "content": content}
        )

    def update(self, comment_id: int, content: str):
        self.conn.execute(
            text("UPDATE Comments SET content = :content WHERE id = :id"),
            {"content": content, "id": comment_id}
        )

    def delete(self, comment_id: int):
        self.conn.execute(text("DELETE FROM Comments WHERE id = :id"), {"id": comment_id})
    
    def get_by_id(self, comment_id: int):
        return self.conn.execute(text("""
            SELECT c.id, c.content, c.created_date, c.user_id, c.post_id
            FROM Comments c
            WHERE c.id = :id
        """), {"id": comment_id}).fetchone()
    
    def get_by_user_id(self, user_id: int):
        return self.conn.execute(text("""
            SELECT c.id, c.content, c.created_date, c.post_id
            FROM Comments c
            WHERE c.user_id = :user_id
            ORDER BY c.created_date DESC
        """), {"user_id": user_id}).fetchall()