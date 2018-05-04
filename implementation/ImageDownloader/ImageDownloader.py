from google_images_download import google_images_download

from config import IMAGE_DOWNLOADER


class ImageDownloader:
    @staticmethod
    def download_images(keyword: str, number=100):
        response = google_images_download.googleimagesdownload()
        response.download({'keywords': keyword, 'limit': number, 'format': 'jpg', 'size': '>400*300', 'type': 'face',
                           'output_directory': IMAGE_DOWNLOADER, 'chromedriver': '/usr/bin/google-chrome'})


if __name__ == '__main__':
    ImageDownloader.download_images("Trump", number=1000)
    ImageDownloader.download_images("Nicolas Cage", number=1000)
