class PostService:
    def __init__(self, post_repo, topic_repo):
        self.post_repo = post_repo
        self.topic_repo = topic_repo

    def create_post_with_topic(self, topic_title: str, content: str, user_id: int):
        topic = self.topic_repo.get_by_title(topic_title)
        if topic:
            topic_id = topic.id
        else:
            topic_id = self.topic_repo.create(title=topic_title, description="", user_id=user_id)
        self.post_repo.create(topic_id=topic_id, user_id=user_id, content=content)
        return topic_id