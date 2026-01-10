from sqlalchemy import text

class BaseRepository:
    def __init__(self, conn):
        self.conn = conn