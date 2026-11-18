from langchain.tools import tool
from models import Profile

def create_portfolio_tools(profile: Profile):
    @tool
    def get_profile() -> dict:
        """Get verified professional background, experience, and skills."""
        return profile.model_dump()
    
    @tool
    def get_resume() -> dict:
        """Request the full resume view when the visitor asks to see it."""
        return {
            "profileId": profile.profileId,
            "action": "show_resume",
        }
    return [get_profile, get_resume];