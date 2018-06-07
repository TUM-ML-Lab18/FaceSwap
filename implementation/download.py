from Utils.Downloader.VideoDownloader import VideoDownloader

if __name__ == '__main__':
    # VideoDownloader.download_video("https://www.youtube.com/watch?v=7E_Mvgsk6PY")  # Kanzlerin Merkel zum #Brexit
    VideoDownloader.download_video("https://www.youtube.com/watch?v=u7srMYGveK8",
                                   path='/nfs/students/summer-term-2018/project_2/data/YT_CAR_DRIVING/raw/images')
    VideoDownloader.download_video("https://www.youtube.com/watch?v=RIXOPcJVTRY",
                                   path='/nfs/students/summer-term-2018/project_2/data/YT_CAR_DRIVING/raw/images')
    VideoDownloader.download_video("https://www.youtube.com/watch?v=wSMDgclRPh4",
                                   path='/nfs/students/summer-term-2018/project_2/data/YT_CAR_DRIVING/raw/images')
    VideoDownloader.download_video("https://www.youtube.com/watch?v=_Ceh-dY5NDA",
                                   path='/nfs/students/summer-term-2018/project_2/data/YT_CAR_DRIVING/raw/images')
    VideoDownloader.download_video("https://www.youtube.com/watch?v=PXnO2IbCnXg&t=360s",
                                   path='/nfs/students/summer-term-2018/project_2/data/YT_CAR_DRIVING/raw/images')
    VideoDownloader.download_video("https://www.youtube.com/watch?v=iK2vJsTuYlQ",
                                   path='/nfs/students/summer-term-2018/project_2/data/YT_CAR_DRIVING/raw/images')
    VideoDownloader.extract_frames(path='/nfs/students/summer-term-2018/project_2/data/YT_CAR_DRIVING/raw/images')
