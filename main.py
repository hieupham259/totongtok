from src.pyktok import pyktok as pyk


def save_video_from(tt_ent: str, ent_type: str='user'):
    """
    Save videos from a TikTok profile.
    tt_ent: TikTok entity (username or hashtag).
    """
    pyk.save_tiktok_multi_page(tt_ent, ent_type=ent_type, save_video=True, video_ct=2, metadata_fn='')

def save_video_from_url(video_url: str):
    """
    Save a video from a TikTok URL.
    video_url: TikTok video URL.
    """
    pyk.save_tiktok(tt_ent='url', video_url=video_url, save_video=True, metadata_fn='')

if __name__ == "__main__":
    # save_video_from('_nguyenanhngocc_', ent_type='user')
    # save_video_from('sinhton', ent_type='hashtag')
    save_video_from_url('https://www.tiktok.com/@lanhxinhyeu06/video/7587769350069832980?lang=en')