from src.pyktok import pyktok as pyk


def save_video_from_profile(tt_ent: str):
    """
    Save videos from a TikTok profile.
    tt_ent: TikTok entity (username).
    """
    pyk.save_tiktok_multi_page(tt_ent, ent_type='user', save_video=True, video_ct=10, metadata_fn='')

if __name__ == "__main__":
    save_video_from_profile('ngh.giang')