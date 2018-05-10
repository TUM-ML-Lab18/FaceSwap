from ImageDownloader.VideoDownloader import VideoDownloader

if __name__ == '__main__':
    VideoDownloader.download_video(
        "https://www.youtube.com/watch?v=YNt37_EQydY")  # Kanzlerin Merkel zu den Anschlägen in Barcelona und Cambrils
    VideoDownloader.download_video(
        "https://www.youtube.com/watch?v=hCHofE6g42c")  # Fonds für Unternehmerinnen - "Mehrwert für Frauen"
    VideoDownloader.download_video("https://www.youtube.com/watch?v=JxPTC6xFPu0")  # G20: Kanzlerin dankt Einsatzkräften
    VideoDownloader.download_video("https://www.youtube.com/watch?v=-IRJvbP0NmU")  # Kanzlerin Merkel vor G20-Gipfel
    VideoDownloader.download_video(
        "https://www.youtube.com/watch?v=-bIBiqoUm4o")  # Kanzlerin Merkel zum Tod von Altkanzler Helmut Kohl
    VideoDownloader.download_video(
        "https://www.youtube.com/watch?v=8yiUJeXrW_Y")  # Weltklimaabkommen - Kanzlerin Merkel zum Pariser Klimaabkommen und dem Ausstieg der USA
    VideoDownloader.download_video("https://www.youtube.com/watch?v=XJF036POvRY")  # Regierungserklärung zum Brexit
    VideoDownloader.extract_frames()