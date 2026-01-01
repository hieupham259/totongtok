from src.pyktok import pyktok as pyk


def save_video_from(tt_ent: str, ent_type: str='user'):
    """
    Save videos from a TikTok profile.
    tt_ent: TikTok entity (username or hashtag).
    """
    pyk.save_tiktok_multi_page(tt_ent, ent_type=ent_type, save_video=True, video_ct=2, metadata_fn='')


if __name__ == "__main__":
    # save_video_from('_nguyenanhngocc_', ent_type='user')
    save_video_from('sinhton', ent_type='hashtag')