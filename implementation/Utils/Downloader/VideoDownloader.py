from pathlib import Path

import cv2
from pytube import YouTube

from Configuration.config_general import VIDEO_DOWNLOADER
from Utils.Logging.LoggingUtils import print_progress_bar


class VideoDownloader:
    @staticmethod
    def download_video(url):
        YouTube(url).streams.filter(progressive=True).order_by('resolution').desc().first().download(
            VIDEO_DOWNLOADER)

    @staticmethod
    def extract_frames(path=VIDEO_DOWNLOADER):
        path = Path(path)
        for video_file in path.iterdir():
            if video_file.is_dir():
                continue
            print(f'Processing video:{video_file.name}')
            folder = path / video_file.name.replace(".mp4", "")
            if folder.exists():
                continue
            folder.mkdir()
            cap = cv2.VideoCapture(video_file.__str__())
            success, image = cap.read()

            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            record_every_nth_frame = 30

            f = int(frames // record_every_nth_frame)

            for i in range(1, f + 1):
                print_progress_bar(i, f)
                cap.set(1, i * record_every_nth_frame - 1)
                success, image = cap.read(1)
                if not success:
                    break
                cv2.imwrite(str(folder / f"{i*record_every_nth_frame}.jpg"), image)

            cap.release()


if __name__ == '__main__':
    # Trump
    VideoDownloader.download_video(
        "https://www.youtube.com/watch?v=Rap1oAvQpy8")  # Trump pulls out of Iran nuclear deal full speech
    VideoDownloader.download_video(
        "https://www.youtube.com/watch?v=aixso4N2vhI")  # President Trump Gives Remarks on the Joint Comprehensive Plan of Action
    VideoDownloader.download_video(
        "https://www.youtube.com/watch?v=ZGe_kruSFEg")  # President Trump Gives Remarks at the National Rifle Association Leadership Forum
    VideoDownloader.download_video(
        "https://www.youtube.com/watch?v=iZ0PA1KnTBE")  # President Trump Delivers an Address to the Nation
    VideoDownloader.download_video(
        "https://www.youtube.com/watch?v=4sVp3yFNEYQ")  # Statement by President Trump on Syria
    VideoDownloader.download_video(
        "https://www.youtube.com/watch?v=daB8VkLTwsc")  # President Trump Speaks at the 2018 House and Senate Republican Member Conference
    VideoDownloader.download_video(
        "https://www.youtube.com/watch?v=wIZo0rqJCVo")  # Weekly Address from President Donald J. Trump: 3/31/2018

    # Merkel
    VideoDownloader.download_video(
        "https://youtu.be/jWtmlIhcouA?t=5")  # Pressekonferenz von Angela Merkel zu möglichen CDU-Ministern am 25.02.18
    VideoDownloader.download_video(
        "https://www.youtube.com/watch?v=kacvLtf7Zcc")  # Neujahrsansprache der Kanzlerin
    VideoDownloader.download_video(
        "https://www.youtube.com/watch?v=npx9IU6v7KY")  # Merkel lädt zum Bürgerdialog über Europas Zukunft ein
    VideoDownloader.download_video(
        "https://www.youtube.com/watch?v=o4dW61UC3wA")  # Merkel: Wir wollen Vollbeschäftigung erreichen
    VideoDownloader.download_video(
        "https://www.youtube.com/watch?v=tkMML24uLoY")  # Merkel zum Girl's Day: Klischeefrei Berufe wählen
    VideoDownloader.download_video(
        "https://www.youtube.com/watch?v=aE0binNnZVI")  # Merkel: Unterschiede zwischen Ost und West ausgleichen
    VideoDownloader.download_video(
        "https://www.youtube.com/watch?v=NOw7IE0UmbU")  # Bundeskanzlerin Angela Merkel zum Weltfrauentag
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
    VideoDownloader.download_video("https://www.youtube.com/watch?v=9jL0yoxcL2s")  # Kanzlerin Merkel zum #Brexit
    VideoDownloader.download_video(
        "https://www.youtube.com/watch?v=-TZBqCnRrEU")  # Kanzlerin Merkel zum Ausgang der US-Präsidentschaftswahl
    VideoDownloader.download_video(
        "https://www.youtube.com/watch?v=CXzKHu2usSA")  # Kanzlerin Merkel zum Tod von Guido Westerwelle
    VideoDownloader.download_video(
        "https://www.youtube.com/watch?v=yaF1CI9lLzI")  # Kanzlerin #Merkel zum Tod von Helmut Schmidt
    VideoDownloader.download_video(
        "https://www.youtube.com/watch?v=tyVmDmZm5Bs")  # Kanzlerin Merkel zum Tod von Hans-Dietrich Genscher
    VideoDownloader.download_video(
        "https://www.youtube.com/watch?v=yzOheh9kvJQ")  # Sommer-Pressekonferenz mit Kanzlerin Merkel Regierungserklärung zum Brexit

    VideoDownloader.download_video("")  #
    # VideoDownloader.extract_frames()
