# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 14:40:48 2019

@author: Owner
"""
import os, os.path
import requests
from apiclient.discovery import build
import curses
import threading
from PIL import Image
from resizeimage import resizeimage

class FetchResizeSave(object):
    """Class with resizing and downloading logic"""

    def __init__(self, developer_key=None, custom_search_cx=None,
                 progressbar_fn=None, progress=False):

        # initialise google api
        self._google_custom_search = GoogleCustomSearch(
            developer_key, custom_search_cx, self)

        self._search_result = list()

        self._stdscr = None
        self._progress = False
        self._chunk_sizes = dict()
        self._terminal_lines = dict()
        self._download_progress = dict()
        self._report_progress = progressbar_fn

        if progressbar_fn:
            # user nserted progressbar fn
            self._progress = True
        else:
            if progress:
                # initialise internal progressbar
                self._progress = True
                self._stdscr = curses.initscr()
                self._report_progress = self.__report_progress

    def search(self, search_params, path_to_dir=False, width=None,
               height=None, cache_discovery=True):
        """Fetched images using Google API and does the download and resize
        if path_to_dir and width and height variables are provided.
        :param search_params: parameters for Google API Search
        :param path_to_dir: path where the images should be downloaded
        :param width: crop width of the images
        :param height: crop height of the images
        :param cache_discovery: whether or not to cache the discovery doc
        :return: None
        """

        i = 0
        threads = list()
        for url in self._google_custom_search.search(
            search_params, cache_discovery
        ):
            # initialise image object
            image = GSImage(self)
            image.url = url

            # set thread safe variables
            self._download_progress[url] = 0
            self._terminal_lines[url] = i
            i += 2

            # set thread with function and arguments
            thread = threading.Thread(
                target=self._download_and_resize,
                args=(path_to_dir, image, width, height)
            )

            # start thread
            thread.start()

            # register thread
            threads.append(thread)

        # wait for all threads to end here
        for thread in threads:
            thread.join()

        if self._progress:
            if self._stdscr:
                curses.endwin()

    def set_chunk_size(self, url, content_size):
        """Set images chunk size according to its size
        :param url: image url
        :param content_size: image size
        :return: None
        """

        self._chunk_sizes[url] = int(int(content_size) / 100) + 1

    def _download_and_resize(self, path_to_dir, image, width, height):
        """Method used for threading
        :param path_to_dir: path to download dir
        :param image: image object
        :param width: crop width
        :param height: crop height
        :return: None
        """

        if path_to_dir:
            image.download(path_to_dir)
            if width and height:
                image.resize(width, height)
        self._search_result.append(image)

    def results(self):
        """Returns objects of downloaded images
        :return: list
        """

        return self._search_result

    def download(self, url, path_to_dir):
        """Downloads image from url to path dir
        Used only by GSImage class
        :param url: image url
        :param path_to_dir: path to directory where image should be saved
        :return: path to image
        """

        if not os.path.exists(path_to_dir):
            os.makedirs(path_to_dir)

        raw_filename = url.split('/')[-1].split('?')[0]
        basename, ext = os.path.splitext(raw_filename)
        filename = "".join(x for x in basename if x.isalnum()) + ext

        path_to_image = os.path.join(path_to_dir, filename)

        with open(path_to_image, 'wb+') as f:
            for chunk in self.get_raw_data(url):
                f.write(chunk)

        return path_to_image

    def get_raw_data(self, url):
        """Generator method for downloading images in chunks
        :param url: url to image
        :return: raw image data
        """

        with requests.get(url, stream=True) as req:
            for chunk in req.iter_content(chunk_size=self._chunk_sizes[url]):

                # filter out keep-alive new chunks
                if chunk:

                    # report progress
                    if self._progress:
                        self._download_progress[url] += 1
                        if self._download_progress[url] <= 100:
                            self._report_progress(url, self._download_progress[url])

                    yield chunk

    @staticmethod
    def resize(path_to_image, width, height):
        """Resize the image and save it again.
        :param path_to_image: os.path
        :param width: int
        :param height: int
        :return: None
        """

        fd_img = open(path_to_image, 'rb')
        img = Image.open(fd_img)
        img = resizeimage.resize_cover(img, [int(width), int(height)])
        img.save(path_to_image, img.format)
        fd_img.close()

    def __report_progress(self, url, progress):
        """Prints a progress bar in terminal
        :param url:
        :param progress:
        :return:
        """

        self._stdscr.addstr(
            self._terminal_lines[url], 0, "Downloading file: {0}".format(url)
        )
        self._stdscr.addstr(
            self._terminal_lines[url] + 1, 0,
            "Progress: [{1:100}] {0}%".format(progress, "#" * progress)
        )
        self._stdscr.refresh()


class GSImage(object):
    """Class for handling one image"""

    def __init__(self, fetch_resize_save):
        self._fetch_resize_save = fetch_resize_save

        self._url = None
        self._path = None

        self.resized = False

    @property
    def url(self):
        """Returns the image url
        :return: url
        """

        return self._url

    @url.setter
    def url(self, image_url):
        """Sets the image url
        :param image_url: url
        :return: None
        """

        self._url = image_url

    @property
    def path(self):
        """Returns image path
        :return: path
        """

        return self._path

    @path.setter
    def path(self, image_path):
        """Sets image path
        :param image_path: path
        :return: None
        """

        self._path = image_path

    def download(self, path_to_dir):
        """Downloads image from url to path
        :param path_to_dir: path
        :return: None
        """

        self._path = self._fetch_resize_save.download(self._url, path_to_dir)

    def get_raw_data(self):
        """Gets images raw data
        :return: raw data
        """

        return b''.join(list(self._fetch_resize_save.get_raw_data(self._url)))

    def copy_to(self, obj, raw_data=None):
        """Copies raw image data to another object, preferably BytesIO
        :param obj: BytesIO
        :param raw_data: raw data
        :return: None
        """

        if not raw_data:
            raw_data = self.get_raw_data()

        obj.write(raw_data)

    def resize(self, width, height):
        """Resize the image
        :param width: int
        :param height: int
        :return: None
        """

        self._fetch_resize_save.__class__.resize(self._path, width, height)
        self.resized = True

class GoogleCustomSearch(object):
    """Wrapper class for Google images search api"""

    def __init__(self, developer_key=None,
                 custom_search_cx=None,
                 fethch_resize_save=None):

        self._developer_key = developer_key or \
                              os.environ.get('GCS_DEVELOPER_KEY')
        self._custom_search_cx = custom_search_cx or \
                                 os.environ.get('GCS_CX')

        self._google_build = None
        self._fethch_resize_save = fethch_resize_save

        self._search_params_keys = {
            'q': None,
            'searchType': 'image',
            'num': 1,
            'start':1,
            'imgType': None,
            'imgSize': None,
            'fileType': None,
            'safe': 'off',
            'imgDominantColor': None
        }

    def _query_google_api(self, search_params, cache_discovery=True):
        """Queries Google api
        :param search_params: dict of params
        :param cache_discovery whether or not to cache the discovery doc
        :return: search result object
        """

        if not self._google_build:
            self._google_build = build("customsearch", "v1",
                                       developerKey=self._developer_key,
                                       cache_discovery=cache_discovery)

        return self._google_build.cse().list(
            cx=self._custom_search_cx, **search_params).execute()

    def _search_params(self, params):
        """Received a dict of params and merges
        it with default params dict
        :param params: dict
        :return: dict
        """

        search_params = {}

        for key, value in self._search_params_keys.items():
            params_value = params.get(key)
            if params_value:
                # take user defined param value if defined
                search_params[key] = params_value
            elif value:
                # take default param value if defined
                search_params[key] = value

        return search_params

    def search(self, params, cache_discovery=True):
        """Search for images and returns
        them using generator object
        :param params: search params
        :param cache_discovery whether or not to cache the discovery doc
        :return: yields url to searched image
        """

        search_params = self._search_params(params)

        res = self._query_google_api(search_params, cache_discovery)

        for image in res.get('items', []):
            try:
                response = requests.head(image['link'], timeout=5)
                content_length = response.headers.get('Content-Length')

                # check if the url is valid
                if response.status_code == 200 and \
                        'image' in response.headers['Content-Type'] and \
                        content_length:

                    # calculate download chunk size based on image size
                    self._fethch_resize_save.set_chunk_size(
                        image['link'], content_length
                    )

                    # if everything is ok, yield image url back
                    yield image['link']

                else:
                    # validation failed, go with another image
                    continue

            except requests.exceptions.ConnectTimeout:
                pass
            except requests.exceptions.SSLError:
                pass


class GoogleBackendException(Exception):
    """Exception handler for search api"""

"""
OUR CODE STARTS HERE
"""

# if you don't enter api key and cx, the package will try to search
# them from environment variables GCS_DEVELOPER_KEY and GCS_CX
gis = FetchResizeSave()

num_per_query = 10
num_images_per_term = 1000
search_terms = ['impressionist self portrait', 'abstract expressionism painting', 'pointilism art']
num_already_searched = []
for term in search_terms:
    term = os.getcwd() + '/data/' + term.replace(' ', '_')
    if os.path.exists(term):   
        num_already_searched.append(len([name for name in os.listdir(term)]))
    else:
        num_already_searched.append(0)

for _query, num_searched in zip(search_terms, num_already_searched):
    print(f'Searching {_query} [{num_searched} - {num_images_per_term}]...')
    for start in range(1, num_images_per_term, num_per_query):
        #define search params:
        _search_params = {
            'q': _query,
            'num': num_per_query,
            'start': start + num_searched,
            #'safe': 'off', #high|medium|off
            #'fileType': 'jpg', #jpg|gif|png
            #'imgType': 'photo', #clipart|face|lineart|news|photo
            'imgSize': 'large',#huge|icon|large|medium|small|xlarge|xxlarge'
            #'imgDominantColor': 'black|blue|brown|gray|green|pink|purple|teal|white|yellow'#black|blue|brown|gray|green|pink|purple|teal|white|yellow
        }
        
        # this will search, download and resize:
        #gis.search(search_params=_search_params, path_to_dir='data/', width=500, height=500)
        
        # search first, then download and resize afterwards
        try:
            gis.search(search_params=_search_params)
            for image in gis.results():
                try:
                    img_dir = _query.replace(' ', "_")
                    image.download(f'data/{img_dir}')
                    image.resize(500, 500)
                except Exception as e:
                    print(e)
        except Exception as e:
            print(e)