from ImageDownloader.VideoDownloader import VideoDownloader

if __name__ == '__main__':
    VideoDownloader.download_video("https://www.youtube.com/watch?v=9jL0yoxcL2s")  # Kanzlerin Merkel zum #Brexit
    VideoDownloader.download_video("https://www.youtube.com/watch?v=-TZBqCnRrEU")  # Kanzlerin Merkel zum Ausgang der US-Präsidentschaftswahl
    VideoDownloader.download_video("https://www.youtube.com/watch?v=CXzKHu2usSA")  # Kanzlerin Merkel zum Tod von Guido Westerwelle
    VideoDownloader.download_video("https://www.youtube.com/watch?v=yaF1CI9lLzI")  # Kanzlerin #Merkel zum Tod von Helmut Schmidt
    VideoDownloader.download_video("https://www.youtube.com/watch?v=tyVmDmZm5Bs")  # Kanzlerin Merkel zum Tod von Hans-Dietrich Genscher
    VideoDownloader.download_video("https://www.youtube.com/watch?v=yzOheh9kvJQ")  # Sommer-Pressekonferenz mit Kanzlerin Merkel Regierungserklärung zum Brexit
    VideoDownloader.extract_frames()