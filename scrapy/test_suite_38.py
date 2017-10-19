import unittest
from scrapy.http import Request
from scrapy.item import BaseItem
from scrapy.utils.spider import iterate_spider_output, iter_spider_classes

from scrapy.spiders import CrawlSpider

import unittest

from scrapy.downloadermiddlewares.ajaxcrawl import AjaxCrawlMiddleware
from scrapy.spiders import Spider
from scrapy.http import Request, HtmlResponse, Response
from scrapy.utils.test import get_crawler

__doctests__ = ['scrapy.downloadermiddlewares.ajaxcrawl']
import unittest

from scrapy.http import Request
from scrapy.downloadermiddlewares.httpauth import HttpAuthMiddleware
from scrapy.spiders import Spider

import unittest
import copy

from scrapy.http import Headers
import hashlib
import tempfile
import unittest
import shutil

from scrapy.dupefilters import RFPDupeFilter
from scrapy.http import Request
from scrapy.utils.python import to_bytes

from unittest import TestCase

from scrapy.downloadermiddlewares.defaultheaders import DefaultHeadersMiddleware
from scrapy.http import Request
from scrapy.spiders import Spider
from scrapy.utils.test import get_crawler
from scrapy.utils.python import to_bytes

from twisted.trial import unittest
from twisted.internet import reactor, defer
from twisted.python.failure import Failure

from scrapy.utils.defer import mustbe_deferred, process_chain, \
    process_chain_both, process_parallel, iter_errback

from six.moves import xrange

import logging
from unittest import TestCase

from testfixtures import LogCapture
from twisted.trial.unittest import TestCase as TrialTestCase
from twisted.internet import defer

from scrapy.utils.test import get_crawler
from tests.mockserver import MockServer
from scrapy.http import Response, Request
from scrapy.spiders import Spider
from scrapy.spidermiddlewares.httperror import HttpErrorMiddleware, HttpError
from scrapy.settings import Settings

import os
import unittest
from six.moves.urllib.parse import urlparse

from scrapy.http import Response, TextResponse, HtmlResponse
from scrapy.utils.python import to_bytes
from scrapy.utils.response import (response_httprepr, open_in_browser,
                                   get_meta_refresh, get_base_url, response_status_message)

__doctests__ = ['scrapy.utils.response']

from twisted.internet import defer
from twisted.trial.unittest import TestCase
from scrapy.utils.test import get_crawler
from tests.spiders import FollowAllSpider, ItemSpider, ErrorSpider
from tests.mockserver import MockServer

import unittest
import warnings
from six.moves import reload_module

import json
import unittest
import datetime
from decimal import Decimal

from twisted.internet import defer

from scrapy.utils.serialize import ScrapyJSONEncoder
from scrapy.http import Request, Response

import gc
import functools
import operator
import unittest
from itertools import count
import platform
import six

from scrapy.utils.python import (
    memoizemethod_noargs, binary_is_text, equal_attributes,
    WeakKeyCache, stringify_dict, get_func_args, to_bytes, to_unicode,
    without_none_values)

__doctests__ = ['scrapy.utils.python']

import unittest

from scrapy.settings import BaseSettings
from scrapy.utils.conf import build_component_list, arglist_to_dict

# -*- coding: utf-8 -*-
import unittest

import six
from six.moves.urllib.parse import urlparse

from scrapy.spiders import Spider
from scrapy.utils.url import (url_is_from_any_domain, url_is_from_spider,
                              add_http_if_no_scheme, guess_scheme,
                              parse_url, strip_url)

__doctests__ = ['scrapy.utils.url']

from unittest import TestCase

from scrapy.spidermiddlewares.urllength import UrlLengthMiddleware
from scrapy.http import Response, Request
from scrapy.spiders import Spider

import os
from datetime import datetime
import shutil
from twisted.trial import unittest

from scrapy.extensions.spiderstate import SpiderState
from scrapy.spiders import Spider
from scrapy.exceptions import NotConfigured
from scrapy.utils.test import get_crawler

from importlib import import_module
from twisted.trial import unittest
import json
import logging

from testfixtures import LogCapture
from twisted.internet import defer
from twisted.trial.unittest import TestCase

from scrapy.http import Request
from scrapy.crawler import CrawlerRunner
from scrapy.utils.python import to_unicode
from tests.spiders import FollowAllSpider, DelaySpider, SimpleSpider, \
    BrokenStartRequestsSpider, SingleRequestSpider, DuplicateStartRequestsSpider
from tests.mockserver import MockServer

import unittest

from scrapy.downloadermiddlewares.downloadtimeout import DownloadTimeoutMiddleware
from scrapy.spiders import Spider
from scrapy.http import Request
from scrapy.utils.test import get_crawler

import gzip
import inspect
import warnings
from io import BytesIO

from testfixtures import LogCapture
from twisted.trial import unittest

from scrapy import signals
from scrapy.settings import Settings
from scrapy.http import Request, Response, TextResponse, XmlResponse, HtmlResponse
from scrapy.spiders.init import InitSpider
from scrapy.spiders import Spider, BaseSpider, CrawlSpider, Rule, XMLFeedSpider, \
    CSVFeedSpider, SitemapSpider
from scrapy.linkextractors import LinkExtractor
from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.utils.trackref import object_ref
from scrapy.utils.test import get_crawler

from tests import mock

from __future__ import print_function
import unittest
from scrapy.http import Request
from scrapy.utils.request import request_fingerprint, _fingerprint_cache, \
    request_authenticate, request_httprepr
import unittest
import os
import tempfile
import shutil
import contextlib
from scrapy.utils.project import data_path

import os
import sys
import subprocess
import tempfile
from time import sleep
from os.path import exists, join, abspath
from shutil import rmtree, copytree
from tempfile import mkdtemp
from contextlib import contextmanager

from twisted.trial import unittest
from twisted.internet import defer

import scrapy
from scrapy.utils.python import to_native_str
from scrapy.utils.python import retry_on_eintr
from scrapy.utils.test import get_testenv
from scrapy.utils.testsite import SiteTest
from scrapy.utils.testproc import ProcessTest

from unittest import TestCase

from scrapy.spiders import Spider
from scrapy.http import Request
from scrapy.downloadermiddlewares.useragent import UserAgentMiddleware
from scrapy.utils.test import get_crawler

# -*- coding: utf-8 -*-
import unittest
from scrapy.linkextractors.regex import RegexLinkExtractor
from scrapy.http import HtmlResponse
from scrapy.link import Link
from scrapy.linkextractors.htmlparser import HtmlParserLinkExtractor
from scrapy.linkextractors.sgml import SgmlLinkExtractor, BaseSgmlLinkExtractor
from tests import get_testdata

from tests.test_linkextractors import Base

from unittest import TextTestResult

from twisted.trial import unittest

from scrapy.spiders import Spider
from scrapy.http import Request
from scrapy.item import Item, Field
from scrapy.contracts import ContractsManager
from scrapy.contracts.default import (
    UrlContract,
    ReturnsContract,
    ScrapesContract,
)

import sys
from twisted.trial import unittest
from twisted.internet import defer

import scrapy
from scrapy.utils.testproc import ProcessTest

from unittest import TestCase, main
from scrapy.http import Response, XmlResponse
from scrapy.downloadermiddlewares.decompression import DecompressionMiddleware
from scrapy.spiders import Spider
from tests import get_testdata
from scrapy.utils.test import assert_samelines

import unittest

from scrapy.utils.console import get_shell_embed_func
try:
    import bpython
    bpy = True
    del bpython
except ImportError:
    bpy = False
try:
    import IPython
    ipy = True
    del IPython
except ImportError:
    ipy = False
import os
import random
import time
import hashlib
import warnings
from tempfile import mkdtemp
from shutil import rmtree
from six.moves.urllib.parse import urlparse
from six import BytesIO

from twisted.trial import unittest
from twisted.internet import defer

from scrapy.pipelines.files import FilesPipeline, FSFilesStore, S3FilesStore, GCSFilesStore
from scrapy.item import Item, Field
from scrapy.http import Request, Response
from scrapy.settings import Settings
from scrapy.utils.python import to_bytes
from scrapy.utils.test import assert_aws_environ, get_s3_content_and_delete
from scrapy.utils.test import assert_gcs_environ, get_gcs_content_and_delete
from scrapy.utils.boto import is_botocore

from tests import mock

from twisted.trial import unittest

from scrapy.settings import Settings
from scrapy.exceptions import NotConfigured
from scrapy.middleware import MiddlewareManager
import six
from __future__ import absolute_import
import re
import json
import marshal
import tempfile
import unittest
from io import BytesIO
from datetime import datetime
from six.moves import cPickle as pickle

import lxml.etree
import six

from scrapy.item import Item, Field
from scrapy.utils.python import to_unicode
from scrapy.exporters import (
    BaseItemExporter, PprintItemExporter, PickleItemExporter, CsvItemExporter,
    XmlItemExporter, JsonLinesItemExporter, JsonItemExporter,
    PythonItemExporter, MarshalItemExporter
)

# -*- coding: utf-8 -*-
from __future__ import absolute_import
import re
from twisted.internet import reactor, error
from twisted.internet.defer import Deferred, DeferredList, maybeDeferred
from twisted.python import failure
from twisted.trial import unittest
from scrapy.downloadermiddlewares.robotstxt import (RobotsTxtMiddleware,
                                                    logger as mw_module_logger)
from scrapy.exceptions import IgnoreRequest, NotConfigured
from scrapy.http import Request, Response, TextResponse
from scrapy.settings import Settings
from tests import mock

import unittest

from scrapy.utils.http import decode_chunked_transfer
import os
import hashlib
import random
import warnings
from tempfile import mkdtemp, TemporaryFile
from shutil import rmtree

from twisted.trial import unittest

from scrapy.item import Item, Field
from scrapy.http import Request, Response
from scrapy.settings import Settings
from scrapy.pipelines.images import ImagesPipeline
from scrapy.utils.python import to_bytes

skip = False
try:
    from PIL import Image
except ImportError as e:
    skip = 'Missing Python Imaging Library, install https://pypi.python.org/pypi/Pillow'
else:
    encoders = set(('jpeg_encoder', 'jpeg_decoder'))
    if not encoders.issubset(set(Image.core.__dict__)):
        skip = 'Missing JPEG encoders'

from six.moves.urllib.parse import urlparse
import unittest

from io import BytesIO
from unittest import TestCase, SkipTest
from os.path import join
from gzip import GzipFile

from scrapy.spiders import Spider
from scrapy.http import Response, Request, HtmlResponse
from scrapy.downloadermiddlewares.httpcompression import HttpCompressionMiddleware, \
    ACCEPTED_ENCODINGS
from scrapy.responsetypes import responsetypes
from scrapy.utils.gz import gunzip
from tests import tests_datadir
from w3lib.encoding import resolve_encoding


SAMPLEDIR = join(tests_datadir, 'compressed')

FORMAT = {
        'gzip': ('html-gzip.bin', 'gzip'),
        'x-gzip': ('html-gzip.bin', 'gzip'),
        'rawdeflate': ('html-rawdeflate.bin', 'deflate'),
        'zlibdeflate': ('html-zlibdeflate.bin', 'deflate'),
        'br': ('html-br.bin', 'br')
        }

import unittest
import warnings
import six

from scrapy.link import Link

import unittest
from os.path import join

from w3lib.encoding import html_to_unicode

from scrapy.utils.gz import gunzip, is_gzipped
from scrapy.http import Response, Headers
from tests import tests_datadir

SAMPLEDIR = join(tests_datadir, 'compressed')

import pickle

from queuelib.tests import test_queue as t
from scrapy.squeues import MarshalFifoDiskQueue, MarshalLifoDiskQueue, PickleFifoDiskQueue, PickleLifoDiskQueue
from scrapy.item import Item, Field
from scrapy.http import Request
from scrapy.loader import ItemLoader
import sys
import unittest

import six

from scrapy.item import ABCMeta, Item, ItemMeta, Field
from tests import mock


PY36_PLUS = (sys.version_info.major >= 3) and (sys.version_info.minor >= 6)

from os.path import join, abspath
from twisted.trial import unittest
from twisted.internet import defer
from scrapy.utils.testsite import SiteTest
from scrapy.utils.testproc import ProcessTest
from scrapy.utils.python import to_native_str
from tests.test_commands import CommandTest

from __future__ import absolute_import
import os
import csv
import json
from io import BytesIO
import tempfile
import shutil
from six.moves.urllib.parse import urlparse

from zope.interface.verify import verifyObject
from twisted.trial import unittest
from twisted.internet import defer
from scrapy.crawler import CrawlerRunner
from scrapy.settings import Settings
from tests.mockserver import MockServer
from w3lib.url import path_to_file_uri

import scrapy
from scrapy.extensions.feedexport import (
    IFeedStorage, FileFeedStorage, FTPFeedStorage,
    S3FeedStorage, StdoutFeedStorage,
    BlockingFeedStorage)
from scrapy.utils.test import assert_aws_environ, get_s3_content_and_delete, get_crawler
from scrapy.utils.python import to_native_str

# -*- coding: utf-8 -*-

import unittest

from scrapy.downloadermiddlewares.redirect import RedirectMiddleware, MetaRefreshMiddleware
from scrapy.spiders import Spider
from scrapy.exceptions import IgnoreRequest
from scrapy.http import Request, Response, HtmlResponse
from scrapy.utils.test import get_crawler

# -*- coding: utf-8 -*-
import cgi
import unittest
import re

import six
from six.moves import xmlrpc_client as xmlrpclib
from six.moves.urllib.parse import urlparse, parse_qs, unquote
if six.PY3:
    from urllib.parse import unquote_to_bytes

from scrapy.http import Request, FormRequest, XmlRpcRequest, Headers, HtmlResponse
from scrapy.utils.python import to_bytes, to_native_str

# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import logging
import unittest

from testfixtures import LogCapture
from twisted.python.failure import Failure

from scrapy.utils.log import (failure_to_exc_info, TopLevelFormatter,
                              LogCounterHandler, StreamLogger)
from scrapy.utils.test import get_crawler

import copy
import unittest
from collections import Mapping, MutableMapping

from scrapy.utils.datatypes import CaselessDict, SequenceExclude

__doctests__ = ['scrapy.utils.datatypes']
from testfixtures import LogCapture
from twisted.trial import unittest
from twisted.python.failure import Failure
from twisted.internet import defer, reactor
from pydispatch import dispatcher

from scrapy.utils.signal import send_catch_log, send_catch_log_deferred

import os
from shutil import rmtree
from tempfile import mkdtemp
import unittest
from scrapy.utils.template import render_templatefile


__doctests__ = ['scrapy.utils.template']

# -*- coding: utf-8 -*-
import unittest

import six
from w3lib.encoding import resolve_encoding

from scrapy.http import (Request, Response, TextResponse, HtmlResponse,
                         XmlResponse, Headers)
from scrapy.selector import Selector
from scrapy.utils.python import to_native_str
from scrapy.exceptions import NotSupported
from scrapy.link import Link
from tests import get_testdata

import re
import logging
from unittest import TestCase
from testfixtures import LogCapture

from scrapy.http import Response, Request
from scrapy.spiders import Spider
from scrapy.utils.test import get_crawler
from scrapy.exceptions import NotConfigured
from scrapy.downloadermiddlewares.cookies import CookiesMiddleware

import unittest
from twisted.internet import defer
from twisted.internet.error import TimeoutError, DNSLookupError, \
        ConnectionRefusedError, ConnectionDone, ConnectError, \
        ConnectionLost, TCPTimedOutError
from twisted.web.client import ResponseFailed

from scrapy.downloadermiddlewares.retry import RetryMiddleware
from scrapy.spiders import Spider
from scrapy.http import Request, Response
from scrapy.utils.test import get_crawler

from six.moves.urllib.parse import urlparse
from unittest import TestCase

from scrapy.http import Request, Response
from scrapy.http.cookies import WrappedRequest, WrappedResponse

import unittest
from six.moves.urllib.parse import urlparse

from scrapy.http import Request
from scrapy.utils.httpobj import urlparse_cached
"""
from twisted.internet import defer
Tests borrowed from the twisted.web.client tests.
"""
import os
import six
import shutil

from twisted.trial import unittest
from twisted.web import server, static, util, resource
from twisted.internet import reactor, defer
from twisted.test.proto_helpers import StringTransport
from twisted.python.filepath import FilePath
from twisted.protocols.policies import WrappingFactory
from twisted.internet.defer import inlineCallbacks

from scrapy.core.downloader import webclient as client
from scrapy.http import Request, Headers
from scrapy.utils.python import to_bytes, to_unicode

import os
import sys
from functools import partial
from twisted.trial.unittest import TestCase, SkipTest

from scrapy.downloadermiddlewares.httpproxy import HttpProxyMiddleware
from scrapy.exceptions import NotConfigured
from scrapy.http import Response, Request
from scrapy.spiders import Spider
from scrapy.crawler import Crawler
from scrapy.settings import Settings

spider = Spider('foo')

from unittest import TestCase

from scrapy.spidermiddlewares.depth import DepthMiddleware
from scrapy.http import Response, Request
from scrapy.spiders import Spider
from scrapy.statscollectors import StatsCollector
from scrapy.utils.test import get_crawler

from __future__ import print_function
from testfixtures import LogCapture
from twisted.trial import unittest
from twisted.python.failure import Failure
from twisted.internet import reactor
from twisted.internet.defer import Deferred, inlineCallbacks

from scrapy.http import Request, Response
from scrapy.settings import Settings
from scrapy.spiders import Spider
from scrapy.utils.request import request_fingerprint
from scrapy.pipelines.media import MediaPipeline
from scrapy.utils.log import failure_to_exc_info
from scrapy.utils.signal import disconnect_all
from scrapy import signals

"""
Scrapy engine tests

This starts a testing web server (using twisted.server.Site) and then crawls it
with the Scrapy crawler.

To view the testing web server in a browser you can start it by running this
module with the ``runserver`` argument::

    python test_engine.py runserver
"""

from __future__ import print_function
import sys, os, re
from six.moves.urllib.parse import urlparse

from twisted.internet import reactor, defer
from twisted.web import server, static, util
from twisted.trial import unittest

from scrapy import signals
from scrapy.core.engine import ExecutionEngine
from scrapy.utils.test import get_crawler
from pydispatch import dispatcher
from tests import tests_datadir
from scrapy.spiders import Spider
from scrapy.item import Item, Field
from scrapy.linkextractors import LinkExtractor
from scrapy.http import Request
from scrapy.utils.signal import disconnect_all

import logging
import os
import tempfile
import warnings
import unittest

import scrapy
from scrapy.crawler import Crawler, CrawlerRunner, CrawlerProcess
from scrapy.settings import Settings, default_settings
from scrapy.spiderloader import SpiderLoader
from scrapy.utils.log import configure_logging, get_scrapy_root_handler
from scrapy.utils.spider import DefaultSpider
from scrapy.utils.misc import load_object
from scrapy.extensions.throttle import AutoThrottle

from six.moves.urllib.parse import urlparse
from unittest import TestCase
import warnings

from scrapy.exceptions import NotConfigured
from scrapy.http import Response, Request
from scrapy.settings import Settings
from scrapy.spiders import Spider
from scrapy.downloadermiddlewares.redirect import RedirectMiddleware
from scrapy.spidermiddlewares.referer import RefererMiddleware, \
    POLICY_NO_REFERRER, POLICY_NO_REFERRER_WHEN_DOWNGRADE, \
    POLICY_SAME_ORIGIN, POLICY_ORIGIN, POLICY_ORIGIN_WHEN_CROSS_ORIGIN, \
    POLICY_SCRAPY_DEFAULT, POLICY_UNSAFE_URL, \
    POLICY_STRICT_ORIGIN, POLICY_STRICT_ORIGIN_WHEN_CROSS_ORIGIN, \
    DefaultReferrerPolicy, \
    NoReferrerPolicy, NoReferrerWhenDowngradePolicy, \
    OriginWhenCrossOriginPolicy, OriginPolicy, \
    StrictOriginWhenCrossOriginPolicy, StrictOriginPolicy, \
    SameOriginPolicy, UnsafeUrlPolicy, ReferrerPolicy

# -*- coding: utf-8 -*-
import unittest
from scrapy.responsetypes import responsetypes

from scrapy.http import Response, TextResponse, XmlResponse, HtmlResponse, Headers
from unittest import TestCase

from scrapy.downloadermiddlewares.stats import DownloaderStats
from scrapy.http import Request, Response
from scrapy.spiders import Spider
from scrapy.utils.test import get_crawler

# -*- coding: utf-8 -*-
import unittest

from scrapy.http import Request, FormRequest
from scrapy.spiders import Spider
from scrapy.utils.reqser import request_to_dict, request_from_dict

import unittest
import six

from scrapy.spiders import Spider
from scrapy.http import Request, Response
from scrapy.item import Item, Field
from scrapy.logformatter import LogFormatter

# -*- coding: utf-8 -*-
import os
import shutil

from testfixtures import LogCapture
from twisted.internet import defer
from twisted.trial.unittest import TestCase
from w3lib.url import add_or_replace_parameter

from scrapy.crawler import CrawlerRunner
from scrapy import signals
from tests.mockserver import MockServer
from tests.spiders import SimpleSpider

import six
import unittest
from scrapy.utils import trackref
from tests import mock

# -*- coding: utf-8 -*-
from __future__ import absolute_import
import inspect
import unittest
import warnings
from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.utils.deprecate import create_deprecated_class, update_classpath

from tests import mock

# -*- coding: utf-8 -*-
import os
import six
from twisted.trial import unittest

from scrapy.utils.iterators import csviter, xmliter, _body_or_str, xmliter_lxml
from scrapy.http import XmlResponse, TextResponse, Response
from tests import get_testdata

FOOBAR_NL = u"foo" + os.linesep + u"bar"

# coding=utf-8

import unittest
from io import BytesIO
from email.charset import Charset

from scrapy.mail import MailSender
from unittest import TestCase
import six
import scrapy

from __future__ import print_function
import time
import tempfile
import shutil
import unittest
import email.utils
from contextlib import contextmanager
import pytest

from scrapy.http import Response, HtmlResponse, Request
from scrapy.spiders import Spider
from scrapy.settings import Settings
from scrapy.exceptions import IgnoreRequest
from scrapy.utils.test import get_crawler
from scrapy.downloadermiddlewares.httpcache import HttpCacheMiddleware

from unittest import TestCase

from six.moves.urllib.parse import urlparse

from scrapy.http import Response, Request
from scrapy.spiders import Spider
from scrapy.spidermiddlewares.offsite import OffsiteMiddleware
from scrapy.utils.test import get_crawler
import unittest
import six
from functools import partial

from scrapy.loader import ItemLoader
from scrapy.loader.processors import Join, Identity, TakeFirst, \
    Compose, MapCompose, SelectJmes
from scrapy.item import Item, Field
from scrapy.selector import Selector
from scrapy.http import HtmlResponse

# test itemsimport unittest

from scrapy.utils.sitemap import Sitemap, sitemap_urls_from_robots
import json
import os
import time

from threading import Thread
from libmproxy import controller, proxy
from netlib import http_auth
from testfixtures import LogCapture

from twisted.internet import defer
from twisted.trial.unittest import TestCase
from scrapy.utils.test import get_crawler
from scrapy.http import Request
from tests.spiders import SimpleSpider, SingleRequestSpider
from tests.mockserver import MockServer

from os.path import join

from twisted.trial import unittest
from twisted.internet import defer

from scrapy.utils.testsite import SiteTest
from scrapy.utils.testproc import ProcessTest

from tests import tests_datadir

import os
import six
import contextlib
import shutil
try:
    from unittest import mock
except ImportError:
    import mock

from twisted.trial import unittest
from twisted.protocols.policies import WrappingFactory
from twisted.python.filepath import FilePath
from twisted.internet import reactor, defer, error
from twisted.web import server, static, util, resource
from twisted.web._newclient import ResponseFailed
from twisted.web.http import _DataLoss
from twisted.web.test.test_webclient import ForeverTakingResource, \
        NoLengthResource, HostHeaderResource, \
        PayloadResource
from twisted.cred import portal, checkers, credentials
from w3lib.url import path_to_file_uri

from scrapy.core.downloader.handlers import DownloadHandlers
from scrapy.core.downloader.handlers.datauri import DataURIDownloadHandler
from scrapy.core.downloader.handlers.file import FileDownloadHandler
from scrapy.core.downloader.handlers.http import HTTPDownloadHandler, HttpDownloadHandler
from scrapy.core.downloader.handlers.http10 import HTTP10DownloadHandler
from scrapy.core.downloader.handlers.http11 import HTTP11DownloadHandler
from scrapy.core.downloader.handlers.s3 import S3DownloadHandler

from scrapy.spiders import Spider
from scrapy.http import Headers, Request
from scrapy.http.response.text import TextResponse
from scrapy.responsetypes import responsetypes
from scrapy.settings import Settings
from scrapy.utils.test import get_crawler, skip_if_no_boto
from scrapy.utils.python import to_bytes
from scrapy.exceptions import NotConfigured

from tests.mockserver import MockServer, ssl_context_factory, Echo
from tests.spiders import SingleRequestSpider
import re
import unittest

import pytest

from scrapy.http import HtmlResponse, XmlResponse
from scrapy.link import Link
from scrapy.linkextractors.lxmlhtml import LxmlLinkExtractor
from tests import get_testdata


# a hack to skip base class tests in pytestfrom twisted.trial.unittest import TestCase
from twisted.python.failure import Failure

from scrapy.http import Request, Response
from scrapy.spiders import Spider
from scrapy.core.downloader.middleware import DownloaderMiddlewareManager
from scrapy.utils.test import get_crawler
from scrapy.utils.python import to_bytes
from tests import mock

import warnings
import weakref
from twisted.trial import unittest
from scrapy.http import TextResponse, HtmlResponse, XmlResponse
from scrapy.selector import Selector
from scrapy.selector.lxmlsel import XmlXPathSelector, HtmlXPathSelector, XPathSelector
from lxml import etree

import unittest

from scrapy.spiders import Spider
from scrapy.statscollectors import StatsCollector, DummyStatsCollector
from scrapy.utils.test import get_crawler

"""
Selector tests for cssselect backend
"""
import warnings
from twisted.trial import unittest
from scrapy.selector.csstranslator import (
    ScrapyHTMLTranslator,
    ScrapyGenericTranslator,
    ScrapyXPathExpr
)

from twisted.trial import unittest
from twisted.internet import defer

from scrapy.utils.testsite import SiteTest
from scrapy.utils.testproc import ProcessTest


class ItemWithFiles(Item):
    file_urls = Field()
    files = Field()


def _create_item_with_files(*files):
    item = ItemWithFiles()
    item['file_urls'] = files
    return item


def _prepare_request_object(item_url):
    return Request(
        item_url,
        meta={'response': Response(item_url, status=200, body=b'data')})


if __name__ == "__main__":
    unittest.main()

class NoParseMethodSpiderTest(unittest.TestCase):

    spider_class = Spider

    def test_undefined_parse_method(self):
        spider = self.spider_class('example.com')
        text = b'Random text'
        resp = TextResponse(url="http://www.example.com/random_url", body=text)

        exc_msg = 'Spider.parse callback is not defined'
        with self.assertRaisesRegexp(NotImplementedError, exc_msg):
            spider.parse(resp)

class ProxyConnectTestCase(TestCase):

    def setUp(self):
        self.mockserver = MockServer()
        self.mockserver.__enter__()
        self._oldenv = os.environ.copy()
        self._proxy = HTTPSProxy(8888)
        self._proxy.start()
        # Wait for the proxy to start.
        time.sleep(1.0)
        os.environ['http_proxy'] = 'http://scrapy:scrapy@localhost:8888'
        os.environ['https_proxy'] = 'http://scrapy:scrapy@localhost:8888'

    def tearDown(self):
        self.mockserver.__exit__(None, None, None)
        self._proxy.shutdown()
        os.environ = self._oldenv

    @defer.inlineCallbacks
    def test_https_connect_tunnel(self):
        crawler = get_crawler(SimpleSpider)
        with LogCapture() as l:
            yield crawler.crawl("https://localhost:8999/status?n=200")
        self._assert_got_response_code(200, l)

    @defer.inlineCallbacks
    def test_https_noconnect(self):
        os.environ['https_proxy'] = 'http://scrapy:scrapy@localhost:8888?noconnect'
        crawler = get_crawler(SimpleSpider)
        with LogCapture() as l:
            yield crawler.crawl("https://localhost:8999/status?n=200")
        self._assert_got_response_code(200, l)
        os.environ['https_proxy'] = 'http://scrapy:scrapy@localhost:8888'

    @defer.inlineCallbacks
    def test_https_connect_tunnel_error(self):
        crawler = get_crawler(SimpleSpider)
        with LogCapture() as l:
            yield crawler.crawl("https://localhost:99999/status?n=200")
        self._assert_got_tunnel_error(l)

    @defer.inlineCallbacks
    def test_https_tunnel_auth_error(self):
        os.environ['https_proxy'] = 'http://wrong:wronger@localhost:8888'
        crawler = get_crawler(SimpleSpider)
        with LogCapture() as l:
            yield crawler.crawl("https://localhost:8999/status?n=200")
        # The proxy returns a 407 error code but it does not reach the client;
        # he just sees a TunnelError.
        self._assert_got_tunnel_error(l)
        os.environ['https_proxy'] = 'http://scrapy:scrapy@localhost:8888'

    @defer.inlineCallbacks
    def test_https_tunnel_without_leak_proxy_authorization_header(self):
        request = Request("https://localhost:8999/echo")
        crawler = get_crawler(SingleRequestSpider)
        with LogCapture() as l:
            yield crawler.crawl(seed=request)
        self._assert_got_response_code(200, l)
        echo = json.loads(crawler.spider.meta['responses'][0].body)
        self.assertTrue('Proxy-Authorization' not in echo['headers'])

    @defer.inlineCallbacks
    def test_https_noconnect_auth_error(self):
        os.environ['https_proxy'] = 'http://wrong:wronger@localhost:8888?noconnect'
        crawler = get_crawler(SimpleSpider)
        with LogCapture() as l:
            yield crawler.crawl("https://localhost:8999/status?n=200")
        self._assert_got_response_code(407, l)

    def _assert_got_response_code(self, code, log):
        print(log)
        self.assertEqual(str(log).count('Crawled (%d)' % code), 1)

    def _assert_got_tunnel_error(self, log):
        print(log)
        self.assertIn('TunnelError', str(log))

class TestS3FilesStore(unittest.TestCase):
    @defer.inlineCallbacks
    def test_persist(self):
        assert_aws_environ()
        uri = os.environ.get('S3_TEST_FILE_URI')
        if not uri:
            raise unittest.SkipTest("No S3 URI available for testing")
        data = b"TestS3FilesStore: \xe2\x98\x83"
        buf = BytesIO(data)
        meta = {'foo': 'bar'}
        path = ''
        store = S3FilesStore(uri)
        yield store.persist_file(
            path, buf, info=None, meta=meta,
            headers={'Content-Type': 'image/png'})
        s = yield store.stat_file(path, info=None)
        self.assertIn('last_modified', s)
        self.assertIn('checksum', s)
        self.assertEqual(s['checksum'], '3187896a9657a28163abb31667df64c8')
        u = urlparse(uri)
        content, key = get_s3_content_and_delete(
            u.hostname, u.path[1:], with_key=True)
        self.assertEqual(content, data)
        if is_botocore():
            self.assertEqual(key['Metadata'], {'foo': 'bar'})
            self.assertEqual(
                key['CacheControl'], S3FilesStore.HEADERS['Cache-Control'])
            self.assertEqual(key['ContentType'], 'image/png')
        else:
            self.assertEqual(key.metadata, {'foo': 'bar'})
            self.assertEqual(
                key.cache_control, S3FilesStore.HEADERS['Cache-Control'])
            self.assertEqual(key.content_type, 'image/png')


class ImageDownloadCrawlTestCase(FileDownloadCrawlTestCase):
    pipeline_class = 'scrapy.pipelines.images.ImagesPipeline'
    store_setting_key = 'IMAGES_STORE'
    media_key = 'images'
    media_urls_key = 'image_urls'

    # somehow checksums for images are different for Python 3.3
    expected_checksums = None

class SomeBaseClass(object):
    pass


class TestSpider(Spider):
    name = 'demo_spider'

    def returns_request(self, response):
        """ method which returns request
        @url http://scrapy.org
        @returns requests 1
        """
        return Request('http://scrapy.org', callback=self.returns_item)

    def returns_item(self, response):
        """ method which returns item
        @url http://scrapy.org
        @returns items 1 1
        """
        return TestItem(url=response.url)

    def returns_dict_item(self, response):
        """ method which returns item
        @url http://scrapy.org
        @returns items 1 1
        """
        return {"url": response.url}

    def returns_fail(self, response):
        """ method which returns item
        @url http://scrapy.org
        @returns items 0 0
        """
        return TestItem(url=response.url)

    def returns_dict_fail(self, response):
        """ method which returns item
        @url http://scrapy.org
        @returns items 0 0
        """
        return {'url': response.url}

    def scrapes_item_ok(self, response):
        """ returns item with name and url
        @url http://scrapy.org
        @returns items 1 1
        @scrapes name url
        """
        return TestItem(name='test', url=response.url)

    def scrapes_dict_item_ok(self, response):
        """ returns item with name and url
        @url http://scrapy.org
        @returns items 1 1
        @scrapes name url
        """
        return {'name': 'test', 'url': response.url}

    def scrapes_item_fail(self, response):
        """ returns item with no name
        @url http://scrapy.org
        @returns items 1 1
        @scrapes name url
        """
        return TestItem(url=response.url)

    def scrapes_dict_item_fail(self, response):
        """ returns item with no name
        @url http://scrapy.org
        @returns items 1 1
        @scrapes name url
        """
        return {'url': response.url}

    def parse_no_url(self, response):
        """ method with no url
        @returns items 1 1
        """
        pass


class BaseResponseTest(unittest.TestCase):

    response_class = Response

    def test_init(self):
        # Response requires url in the consturctor
        self.assertRaises(Exception, self.response_class)
        self.assertTrue(isinstance(self.response_class('http://example.com/'), self.response_class))
        if not six.PY2:
            self.assertRaises(TypeError, self.response_class, b"http://example.com")
        # body can be str or None
        self.assertTrue(isinstance(self.response_class('http://example.com/', body=b''), self.response_class))
        self.assertTrue(isinstance(self.response_class('http://example.com/', body=b'body'), self.response_class))
        # test presence of all optional parameters
        self.assertTrue(isinstance(self.response_class('http://example.com/', body=b'', headers={}, status=200), self.response_class))

        r = self.response_class("http://www.example.com")
        assert isinstance(r.url, str)
        self.assertEqual(r.url, "http://www.example.com")
        self.assertEqual(r.status, 200)

        assert isinstance(r.headers, Headers)
        self.assertEqual(r.headers, {})

        headers = {"foo": "bar"}
        body = b"a body"
        r = self.response_class("http://www.example.com", headers=headers, body=body)

        assert r.headers is not headers
        self.assertEqual(r.headers[b"foo"], b"bar")

        r = self.response_class("http://www.example.com", status=301)
        self.assertEqual(r.status, 301)
        r = self.response_class("http://www.example.com", status='301')
        self.assertEqual(r.status, 301)
        self.assertRaises(ValueError, self.response_class, "http://example.com", status='lala200')

    def test_copy(self):
        """Test Response copy"""

        r1 = self.response_class("http://www.example.com", body=b"Some body")
        r1.flags.append('cached')
        r2 = r1.copy()

        self.assertEqual(r1.status, r2.status)
        self.assertEqual(r1.body, r2.body)

        # make sure flags list is shallow copied
        assert r1.flags is not r2.flags, "flags must be a shallow copy, not identical"
        self.assertEqual(r1.flags, r2.flags)

        # make sure headers attribute is shallow copied
        assert r1.headers is not r2.headers, "headers must be a shallow copy, not identical"
        self.assertEqual(r1.headers, r2.headers)

    def test_copy_meta(self):
        req = Request("http://www.example.com")
        req.meta['foo'] = 'bar'
        r1 = self.response_class("http://www.example.com", body=b"Some body", request=req)
        assert r1.meta is req.meta

    def test_copy_inherited_classes(self):
        """Test Response children copies preserve their class"""

        class CustomResponse(self.response_class):
            pass

        r1 = CustomResponse('http://www.example.com')
        r2 = r1.copy()

        assert type(r2) is CustomResponse

    def test_replace(self):
        """Test Response.replace() method"""
        hdrs = Headers({"key": "value"})
        r1 = self.response_class("http://www.example.com")
        r2 = r1.replace(status=301, body=b"New body", headers=hdrs)
        assert r1.body == b''
        self.assertEqual(r1.url, r2.url)
        self.assertEqual((r1.status, r2.status), (200, 301))
        self.assertEqual((r1.body, r2.body), (b'', b"New body"))
        self.assertEqual((r1.headers, r2.headers), ({}, hdrs))

        # Empty attributes (which may fail if not compared properly)
        r3 = self.response_class("http://www.example.com", flags=['cached'])
        r4 = r3.replace(body=b'', flags=[])
        self.assertEqual(r4.body, b'')
        self.assertEqual(r4.flags, [])

    def _assert_response_values(self, response, encoding, body):
        if isinstance(body, six.text_type):
            body_unicode = body
            body_bytes = body.encode(encoding)
        else:
            body_unicode = body.decode(encoding)
            body_bytes = body

        assert isinstance(response.body, bytes)
        assert isinstance(response.text, six.text_type)
        self._assert_response_encoding(response, encoding)
        self.assertEqual(response.body, body_bytes)
        self.assertEqual(response.body_as_unicode(), body_unicode)
        self.assertEqual(response.text, body_unicode)

    def _assert_response_encoding(self, response, encoding):
        self.assertEqual(response.encoding, resolve_encoding(encoding))

    def test_immutable_attributes(self):
        r = self.response_class("http://example.com")
        self.assertRaises(AttributeError, setattr, r, 'url', 'http://example2.com')
        self.assertRaises(AttributeError, setattr, r, 'body', 'xxx')

    def test_urljoin(self):
        """Test urljoin shortcut (only for existence, since behavior equals urljoin)"""
        joined = self.response_class('http://www.example.com').urljoin('/test')
        absolute = 'http://www.example.com/test'
        self.assertEqual(joined, absolute)

    def test_shortcut_attributes(self):
        r = self.response_class("http://example.com", body=b'hello')
        if self.response_class == Response:
            msg = "Response content isn't text"
            self.assertRaisesRegexp(AttributeError, msg, getattr, r, 'text')
            self.assertRaisesRegexp(NotSupported, msg, r.css, 'body')
            self.assertRaisesRegexp(NotSupported, msg, r.xpath, '//body')
        else:
            r.text
            r.css('body')
            r.xpath('//body')

    def test_follow_url_absolute(self):
        self._assert_followed_url('http://foo.example.com',
                                  'http://foo.example.com')

    def test_follow_url_relative(self):
        self._assert_followed_url('foo',
                                  'http://example.com/foo')

    def test_follow_link(self):
        self._assert_followed_url(Link('http://example.com/foo'),
                                  'http://example.com/foo')

    def test_follow_whitespace_url(self):
        self._assert_followed_url('foo ',
                                  'http://example.com/foo%20')

    def test_follow_whitespace_link(self):
        self._assert_followed_url(Link('http://example.com/foo '),
                                  'http://example.com/foo%20')
    def _assert_followed_url(self, follow_obj, target_url, response=None):
        if response is None:
            response = self._links_response()
        req = response.follow(follow_obj)
        self.assertEqual(req.url, target_url)
        return req

    def _links_response(self):
        body = get_testdata('link_extractor', 'sgml_linkextractor.html')
        resp = self.response_class('http://example.com/index', body=body)
        return resp


class FilesystemStorageTest(DefaultStorageTest):

    storage_class = 'scrapy.extensions.httpcache.FilesystemCacheStorage'

class TestItem(NameItem):
    url = Field()
    summary = Field()


class WrappedResponseTest(TestCase):

    def setUp(self):
        self.response = Response("http://www.example.com/page.html",
                                 headers={"Content-TYpe": "text/html"})
        self.wrapped = WrappedResponse(self.response)

    def test_info(self):
        self.assertIs(self.wrapped.info(), self.wrapped)

    def test_getheaders(self):
        self.assertEqual(self.wrapped.getheaders('content-type'), ['text/html'])

    def test_get_all(self):
        # get_all result must be native string
        self.assertEqual(self.wrapped.get_all('content-type'), ['text/html'])

class XmlItemExporterTest(BaseItemExporterTest):

    def _get_exporter(self, **kwargs):
        return XmlItemExporter(self.output, **kwargs)

    def assertXmlEquivalent(self, first, second, msg=None):
        def xmltuple(elem):
            children = list(elem.iterchildren())
            if children:
                return [(child.tag, sorted(xmltuple(child)))
                        for child in children]
            else:
                return [(elem.tag, [(elem.text, ())])]
        def xmlsplit(xmlcontent):
            doc = lxml.etree.fromstring(xmlcontent)
            return xmltuple(doc)
        return self.assertEqual(xmlsplit(first), xmlsplit(second), msg)

    def assertExportResult(self, item, expected_value):
        fp = BytesIO()
        ie = XmlItemExporter(fp)
        ie.start_exporting()
        ie.export_item(item)
        ie.finish_exporting()
        self.assertXmlEquivalent(fp.getvalue(), expected_value)

    def _check_output(self):
        expected_value = b'<?xml version="1.0" encoding="utf-8"?>\n<items><item><age>22</age><name>John\xc2\xa3</name></item></items>'
        self.assertXmlEquivalent(self.output.getvalue(), expected_value)

    def test_multivalued_fields(self):
        self.assertExportResult(
            TestItem(name=[u'John\xa3', u'Doe']),
            b'<?xml version="1.0" encoding="utf-8"?>\n<items><item><name><value>John\xc2\xa3</value><value>Doe</value></name></item></items>'
        )

    def test_nested_item(self):
        i1 = TestItem(name=u'foo\xa3hoo', age='22')
        i2 = dict(name=u'bar', age=i1)
        i3 = TestItem(name=u'buz', age=i2)

        self.assertExportResult(i3,
            b'<?xml version="1.0" encoding="utf-8"?>\n'
            b'<items>'
                b'<item>'
                    b'<age>'
                        b'<age>'
                            b'<age>22</age>'
                            b'<name>foo\xc2\xa3hoo</name>'
                        b'</age>'
                        b'<name>bar</name>'
                    b'</age>'
                    b'<name>buz</name>'
                b'</item>'
            b'</items>'
        )

    def test_nested_list_item(self):
        i1 = TestItem(name=u'foo')
        i2 = dict(name=u'bar', v2={"egg": ["spam"]})
        i3 = TestItem(name=u'buz', age=[i1, i2])

        self.assertExportResult(i3,
            b'<?xml version="1.0" encoding="utf-8"?>\n'
            b'<items>'
                b'<item>'
                    b'<age>'
                        b'<value><name>foo</name></value>'
                        b'<value><name>bar</name><v2><egg><value>spam</value></egg></v2></value>'
                    b'</age>'
                    b'<name>buz</name>'
                b'</item>'
            b'</items>'
        )

    def test_nonstring_types_item(self):
        item = self._get_nonstring_types_item()
        self.assertExportResult(item,
            b'<?xml version="1.0" encoding="utf-8"?>\n'
            b'<items>'
               b'<item>'
                   b'<float>3.14</float>'
                   b'<boolean>False</boolean>'
                   b'<number>22</number>'
                   b'<time>2015-01-01 01:01:01</time>'
               b'</item>'
            b'</items>'
        )


class LogCounterHandlerTest(unittest.TestCase):

    def setUp(self):
        self.logger = logging.getLogger('test')
        self.logger.setLevel(logging.NOTSET)
        self.logger.propagate = False
        self.crawler = get_crawler(settings_dict={'LOG_LEVEL': 'WARNING'})
        self.handler = LogCounterHandler(self.crawler)
        self.logger.addHandler(self.handler)

    def tearDown(self):
        self.logger.propagate = True
        self.logger.removeHandler(self.handler)

    def test_init(self):
        self.assertIsNone(self.crawler.stats.get_value('log_count/DEBUG'))
        self.assertIsNone(self.crawler.stats.get_value('log_count/INFO'))
        self.assertIsNone(self.crawler.stats.get_value('log_count/WARNING'))
        self.assertIsNone(self.crawler.stats.get_value('log_count/ERROR'))
        self.assertIsNone(self.crawler.stats.get_value('log_count/CRITICAL'))

    def test_accepted_level(self):
        self.logger.error('test log msg')
        self.assertEqual(self.crawler.stats.get_value('log_count/ERROR'), 1)

    def test_filtered_out_level(self):
        self.logger.debug('test log msg')
        self.assertIsNone(self.crawler.stats.get_value('log_count/INFO'))


class TestSettingsSameOrigin(MixinSameOrigin, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.SameOriginPolicy'}


class PickleLifoDiskQueueTest(MarshalLifoDiskQueueTest):

    def queue(self):
        return PickleLifoDiskQueue(self.qpath)

    def test_serialize_item(self):
        q = self.queue()
        i = TestItem(name='foo')
        q.push(i)
        i2 = q.pop()
        assert isinstance(i2, TestItem)
        self.assertEqual(i, i2)

    def test_serialize_loader(self):
        q = self.queue()
        l = TestLoader()
        q.push(l)
        l2 = q.pop()
        assert isinstance(l2, TestLoader)
        assert l2.default_item_class is TestItem
        self.assertEqual(l2.name_out('x'), 'xx')

    def test_serialize_request_recursive(self):
        q = self.queue()
        r = Request('http://www.example.com')
        r.meta['request'] = r
        q.push(r)
        r2 = q.pop()
        assert isinstance(r2, Request)
        self.assertEqual(r.url, r2.url)
        assert r2.meta['request'] is r2

class DummyPolicyTest(_BaseTest):

    policy_class = 'scrapy.extensions.httpcache.DummyPolicy'

    def test_middleware(self):
        with self._middleware() as mw:
            assert mw.process_request(self.request, self.spider) is None
            mw.process_response(self.request, self.response, self.spider)
            response = mw.process_request(self.request, self.spider)
            assert isinstance(response, HtmlResponse)
            self.assertEqualResponse(self.response, response)
            assert 'cached' in response.flags

    def test_different_request_response_urls(self):
        with self._middleware() as mw:
            req = Request('http://host.com/path')
            res = Response('http://host2.net/test.html')
            assert mw.process_request(req, self.spider) is None
            mw.process_response(req, res, self.spider)
            cached = mw.process_request(req, self.spider)
            assert isinstance(cached, Response)
            self.assertEqualResponse(res, cached)
            assert 'cached' in cached.flags

    def test_middleware_ignore_missing(self):
        with self._middleware(HTTPCACHE_IGNORE_MISSING=True) as mw:
            self.assertRaises(IgnoreRequest, mw.process_request, self.request, self.spider)
            mw.process_response(self.request, self.response, self.spider)
            response = mw.process_request(self.request, self.spider)
            assert isinstance(response, HtmlResponse)
            self.assertEqualResponse(self.response, response)
            assert 'cached' in response.flags

    def test_middleware_ignore_schemes(self):
        # http responses are cached by default
        req, res = Request('http://test.com/'), Response('http://test.com/')
        with self._middleware() as mw:
            assert mw.process_request(req, self.spider) is None
            mw.process_response(req, res, self.spider)

            cached = mw.process_request(req, self.spider)
            assert isinstance(cached, Response), type(cached)
            self.assertEqualResponse(res, cached)
            assert 'cached' in cached.flags

        # file response is not cached by default
        req, res = Request('file:///tmp/t.txt'), Response('file:///tmp/t.txt')
        with self._middleware() as mw:
            assert mw.process_request(req, self.spider) is None
            mw.process_response(req, res, self.spider)

            assert mw.storage.retrieve_response(self.spider, req) is None
            assert mw.process_request(req, self.spider) is None

        # s3 scheme response is cached by default
        req, res = Request('s3://bucket/key'), Response('http://bucket/key')
        with self._middleware() as mw:
            assert mw.process_request(req, self.spider) is None
            mw.process_response(req, res, self.spider)

            cached = mw.process_request(req, self.spider)
            assert isinstance(cached, Response), type(cached)
            self.assertEqualResponse(res, cached)
            assert 'cached' in cached.flags

        # ignore s3 scheme
        req, res = Request('s3://bucket/key2'), Response('http://bucket/key2')
        with self._middleware(HTTPCACHE_IGNORE_SCHEMES=['s3']) as mw:
            assert mw.process_request(req, self.spider) is None
            mw.process_response(req, res, self.spider)

            assert mw.storage.retrieve_response(self.spider, req) is None
            assert mw.process_request(req, self.spider) is None

    def test_middleware_ignore_http_codes(self):
        # test response is not cached
        with self._middleware(HTTPCACHE_IGNORE_HTTP_CODES=[202]) as mw:
            assert mw.process_request(self.request, self.spider) is None
            mw.process_response(self.request, self.response, self.spider)

            assert mw.storage.retrieve_response(self.spider, self.request) is None
            assert mw.process_request(self.request, self.spider) is None

        # test response is cached
        with self._middleware(HTTPCACHE_IGNORE_HTTP_CODES=[203]) as mw:
            mw.process_response(self.request, self.response, self.spider)
            response = mw.process_request(self.request, self.spider)
            assert isinstance(response, HtmlResponse)
            self.assertEqualResponse(self.response, response)
            assert 'cached' in response.flags


class TestItemLoader(NameItemLoader):
    name_in = MapCompose(lambda v: v.title())


class TestRefererMiddlewareDefault(MixinDefault, TestRefererMiddleware):
    pass


# --- Tests using settings to set policy using class path
class MustbeDeferredTest(unittest.TestCase):
    def test_success_function(self):
        steps = []
        def _append(v):
            steps.append(v)
            return steps

        dfd = mustbe_deferred(_append, 1)
        dfd.addCallback(self.assertEqual, [1, 2]) # it is [1] with maybeDeferred
        steps.append(2) # add another value, that should be catched by assertEqual
        return dfd

    def test_unfired_deferred(self):
        steps = []
        def _append(v):
            steps.append(v)
            dfd = defer.Deferred()
            reactor.callLater(0, dfd.callback, steps)
            return dfd

        dfd = mustbe_deferred(_append, 1)
        dfd.addCallback(self.assertEqual, [1, 2]) # it is [1] with maybeDeferred
        steps.append(2) # add another value, that should be catched by assertEqual
        return dfd

def cb1(value, arg1, arg2):
    return "(cb1 %s %s %s)" % (value, arg1, arg2)
def cb2(value, arg1, arg2):
    return defer.succeed("(cb2 %s %s %s)" % (value, arg1, arg2))
def cb3(value, arg1, arg2):
    return "(cb3 %s %s %s)" % (value, arg1, arg2)
def cb_fail(value, arg1, arg2):
    return Failure(TypeError())
def eb1(failure, arg1, arg2):
    return "(eb1 %s %s %s)" % (failure.value.__class__.__name__, arg1, arg2)


class MaxRetryTimesTest(unittest.TestCase):
    def setUp(self):
        self.crawler = get_crawler(Spider)
        self.spider = self.crawler._create_spider('foo')
        self.mw = RetryMiddleware.from_crawler(self.crawler)
        self.mw.max_retry_times = 2
        self.invalid_url = 'http://www.scrapytest.org/invalid_url'

    def test_with_settings_zero(self):

        # SETTINGS: RETRY_TIMES = 0
        self.mw.max_retry_times = 0

        req = Request(self.invalid_url)
        self._test_retry(req, DNSLookupError('foo'), self.mw.max_retry_times)

    def test_with_metakey_zero(self):

        # SETTINGS: meta(max_retry_times) = 0
        meta_max_retry_times = 0
        
        req = Request(self.invalid_url, meta={'max_retry_times': meta_max_retry_times})
        self._test_retry(req, DNSLookupError('foo'), meta_max_retry_times)

    def test_without_metakey(self):

        # SETTINGS: RETRY_TIMES is NON-ZERO
        self.mw.max_retry_times = 5

        req = Request(self.invalid_url)
        self._test_retry(req, DNSLookupError('foo'), self.mw.max_retry_times)

    def test_with_metakey_greater(self):
        
        # SETINGS: RETRY_TIMES < meta(max_retry_times)
        self.mw.max_retry_times = 2
        meta_max_retry_times = 3

        req1 = Request(self.invalid_url, meta={'max_retry_times': meta_max_retry_times})
        req2 = Request(self.invalid_url)

        self._test_retry(req1, DNSLookupError('foo'), meta_max_retry_times)
        self._test_retry(req2, DNSLookupError('foo'), self.mw.max_retry_times)

    def test_with_metakey_lesser(self):
        
        # SETINGS: RETRY_TIMES > meta(max_retry_times)
        self.mw.max_retry_times = 5
        meta_max_retry_times = 4

        req1 = Request(self.invalid_url, meta={'max_retry_times': meta_max_retry_times})
        req2 = Request(self.invalid_url)

        self._test_retry(req1, DNSLookupError('foo'), meta_max_retry_times)
        self._test_retry(req2, DNSLookupError('foo'), self.mw.max_retry_times)

    def test_with_dont_retry(self):

        # SETTINGS: meta(max_retry_times) = 4
        meta_max_retry_times = 4

        req = Request(self.invalid_url, meta= \
            {'max_retry_times': meta_max_retry_times, 'dont_retry': True})

        self._test_retry(req, DNSLookupError('foo'), 0)


    def _test_retry(self, req, exception, max_retry_times):
        
        for i in range(0, max_retry_times):
            req = self.mw.process_exception(req, exception, self.spider)
            assert isinstance(req, Request)

        # discard it
        req = self.mw.process_exception(req, exception, self.spider)
        self.assertEqual(req, None)


if __name__ == "__main__":
    unittest.main()

class PickleItemExporterTest(BaseItemExporterTest):

    def _get_exporter(self, **kwargs):
        return PickleItemExporter(self.output, **kwargs)

    def _check_output(self):
        self._assert_expected_item(pickle.loads(self.output.getvalue()))

    def test_export_multiple_items(self):
        i1 = TestItem(name='hello', age='world')
        i2 = TestItem(name='bye', age='world')
        f = BytesIO()
        ie = PickleItemExporter(f)
        ie.start_exporting()
        ie.export_item(i1)
        ie.export_item(i2)
        ie.finish_exporting()
        f.seek(0)
        self.assertEqual(pickle.load(f), i1)
        self.assertEqual(pickle.load(f), i2)

    def test_nonstring_types_item(self):
        item = self._get_nonstring_types_item()
        fp = BytesIO()
        ie = PickleItemExporter(fp)
        ie.start_exporting()
        ie.export_item(item)
        ie.finish_exporting()
        self.assertEqual(pickle.loads(fp.getvalue()), item)


class SelectortemLoaderTest(unittest.TestCase):
    response = HtmlResponse(url="", encoding='utf-8', body=b"""
    <html>
    <body>
    <div id="id">marta</div>
    <p>paragraph</p>
    <a href="http://www.scrapy.org">homepage</a>
    <img src="/images/logo.png" width="244" height="65" alt="Scrapy">
    </body>
    </html>
    """)

    def test_constructor(self):
        l = TestItemLoader()
        self.assertEqual(l.selector, None)

    def test_constructor_errors(self):
        l = TestItemLoader()
        self.assertRaises(RuntimeError, l.add_xpath, 'url', '//a/@href')
        self.assertRaises(RuntimeError, l.replace_xpath, 'url', '//a/@href')
        self.assertRaises(RuntimeError, l.get_xpath, '//a/@href')
        self.assertRaises(RuntimeError, l.add_css, 'name', '#name::text')
        self.assertRaises(RuntimeError, l.replace_css, 'name', '#name::text')
        self.assertRaises(RuntimeError, l.get_css, '#name::text')

    def test_constructor_with_selector(self):
        sel = Selector(text=u"<html><body><div>marta</div></body></html>")
        l = TestItemLoader(selector=sel)
        self.assertIs(l.selector, sel)

        l.add_xpath('name', '//div/text()')
        self.assertEqual(l.get_output_value('name'), [u'Marta'])

    def test_constructor_with_selector_css(self):
        sel = Selector(text=u"<html><body><div>marta</div></body></html>")
        l = TestItemLoader(selector=sel)
        self.assertIs(l.selector, sel)

        l.add_css('name', 'div::text')
        self.assertEqual(l.get_output_value('name'), [u'Marta'])

    def test_constructor_with_response(self):
        l = TestItemLoader(response=self.response)
        self.assertTrue(l.selector)

        l.add_xpath('name', '//div/text()')
        self.assertEqual(l.get_output_value('name'), [u'Marta'])

    def test_constructor_with_response_css(self):
        l = TestItemLoader(response=self.response)
        self.assertTrue(l.selector)

        l.add_css('name', 'div::text')
        self.assertEqual(l.get_output_value('name'), [u'Marta'])

        l.add_css('url', 'a::attr(href)')
        self.assertEqual(l.get_output_value('url'), [u'http://www.scrapy.org'])

        # combining/accumulating CSS selectors and XPath expressions
        l.add_xpath('name', '//div/text()')
        self.assertEqual(l.get_output_value('name'), [u'Marta', u'Marta'])

        l.add_xpath('url', '//img/@src')
        self.assertEqual(l.get_output_value('url'), [u'http://www.scrapy.org', u'/images/logo.png'])

    def test_add_xpath_re(self):
        l = TestItemLoader(response=self.response)
        l.add_xpath('name', '//div/text()', re='ma')
        self.assertEqual(l.get_output_value('name'), [u'Ma'])

    def test_replace_xpath(self):
        l = TestItemLoader(response=self.response)
        self.assertTrue(l.selector)
        l.add_xpath('name', '//div/text()')
        self.assertEqual(l.get_output_value('name'), [u'Marta'])
        l.replace_xpath('name', '//p/text()')
        self.assertEqual(l.get_output_value('name'), [u'Paragraph'])

        l.replace_xpath('name', ['//p/text()', '//div/text()'])
        self.assertEqual(l.get_output_value('name'), [u'Paragraph', 'Marta'])

    def test_get_xpath(self):
        l = TestItemLoader(response=self.response)
        self.assertEqual(l.get_xpath('//p/text()'), [u'paragraph'])
        self.assertEqual(l.get_xpath('//p/text()', TakeFirst()), u'paragraph')
        self.assertEqual(l.get_xpath('//p/text()', TakeFirst(), re='pa'), u'pa')

        self.assertEqual(l.get_xpath(['//p/text()', '//div/text()']), [u'paragraph', 'marta'])

    def test_replace_xpath_multi_fields(self):
        l = TestItemLoader(response=self.response)
        l.add_xpath(None, '//div/text()', TakeFirst(), lambda x: {'name': x})
        self.assertEqual(l.get_output_value('name'), [u'Marta'])
        l.replace_xpath(None, '//p/text()', TakeFirst(), lambda x: {'name': x})
        self.assertEqual(l.get_output_value('name'), [u'Paragraph'])

    def test_replace_xpath_re(self):
        l = TestItemLoader(response=self.response)
        self.assertTrue(l.selector)
        l.add_xpath('name', '//div/text()')
        self.assertEqual(l.get_output_value('name'), [u'Marta'])
        l.replace_xpath('name', '//div/text()', re='ma')
        self.assertEqual(l.get_output_value('name'), [u'Ma'])

    def test_add_css_re(self):
        l = TestItemLoader(response=self.response)
        l.add_css('name', 'div::text', re='ma')
        self.assertEqual(l.get_output_value('name'), [u'Ma'])

        l.add_css('url', 'a::attr(href)', re='http://(.+)')
        self.assertEqual(l.get_output_value('url'), [u'www.scrapy.org'])

    def test_replace_css(self):
        l = TestItemLoader(response=self.response)
        self.assertTrue(l.selector)
        l.add_css('name', 'div::text')
        self.assertEqual(l.get_output_value('name'), [u'Marta'])
        l.replace_css('name', 'p::text')
        self.assertEqual(l.get_output_value('name'), [u'Paragraph'])

        l.replace_css('name', ['p::text', 'div::text'])
        self.assertEqual(l.get_output_value('name'), [u'Paragraph', 'Marta'])

        l.add_css('url', 'a::attr(href)', re='http://(.+)')
        self.assertEqual(l.get_output_value('url'), [u'www.scrapy.org'])
        l.replace_css('url', 'img::attr(src)')
        self.assertEqual(l.get_output_value('url'), [u'/images/logo.png'])

    def test_get_css(self):
        l = TestItemLoader(response=self.response)
        self.assertEqual(l.get_css('p::text'), [u'paragraph'])
        self.assertEqual(l.get_css('p::text', TakeFirst()), u'paragraph')
        self.assertEqual(l.get_css('p::text', TakeFirst(), re='pa'), u'pa')

        self.assertEqual(l.get_css(['p::text', 'div::text']), [u'paragraph', 'marta'])
        self.assertEqual(l.get_css(['a::attr(href)', 'img::attr(src)']),
            [u'http://www.scrapy.org', u'/images/logo.png'])

    def test_replace_css_multi_fields(self):
        l = TestItemLoader(response=self.response)
        l.add_css(None, 'div::text', TakeFirst(), lambda x: {'name': x})
        self.assertEqual(l.get_output_value('name'), [u'Marta'])
        l.replace_css(None, 'p::text', TakeFirst(), lambda x: {'name': x})
        self.assertEqual(l.get_output_value('name'), [u'Paragraph'])

        l.add_css(None, 'a::attr(href)', TakeFirst(), lambda x: {'url': x})
        self.assertEqual(l.get_output_value('url'), [u'http://www.scrapy.org'])
        l.replace_css(None, 'img::attr(src)', TakeFirst(), lambda x: {'url': x})
        self.assertEqual(l.get_output_value('url'), [u'/images/logo.png'])

    def test_replace_css_re(self):
        l = TestItemLoader(response=self.response)
        self.assertTrue(l.selector)
        l.add_css('url', 'a::attr(href)')
        self.assertEqual(l.get_output_value('url'), [u'http://www.scrapy.org'])
        l.replace_css('url', 'a::attr(href)', re='http://www\.(.+)')
        self.assertEqual(l.get_output_value('url'), [u'scrapy.org'])


class HTTPSProxy(controller.Master, Thread):

    def __init__(self, port):
        password_manager = http_auth.PassManSingleUser('scrapy', 'scrapy')
        authenticator = http_auth.BasicProxyAuth(password_manager, "mitmproxy")
        cert_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
            'keys', 'mitmproxy-ca.pem')
        server = proxy.ProxyServer(proxy.ProxyConfig(
            authenticator = authenticator,
            cacert = cert_path),
            port)
        Thread.__init__(self)
        controller.Master.__init__(self, server)


class S3TestCase(unittest.TestCase):
    download_handler_cls = S3DownloadHandler

    # test use same example keys than amazon developer guide
    # http://s3.amazonaws.com/awsdocs/S3/20060301/s3-dg-20060301.pdf
    # and the tests described here are the examples from that manual

    AWS_ACCESS_KEY_ID = '0PN5J17HBGZHT7JJ3X82'
    AWS_SECRET_ACCESS_KEY = 'uV3F3YluFJax1cknvbcGwgjvx4QpvB+leU8dUj2o'

    def setUp(self):
        skip_if_no_boto()
        s3reqh = S3DownloadHandler(Settings(), self.AWS_ACCESS_KEY_ID,
                self.AWS_SECRET_ACCESS_KEY,
                httpdownloadhandler=HttpDownloadHandlerMock)
        self.download_request = s3reqh.download_request
        self.spider = Spider('foo')

    @contextlib.contextmanager
    def _mocked_date(self, date):
        try:
            import botocore.auth
        except ImportError:
            yield
        else:
            # We need to mock botocore.auth.formatdate, because otherwise
            # botocore overrides Date header with current date and time
            # and Authorization header is different each time
            with mock.patch('botocore.auth.formatdate') as mock_formatdate:
                mock_formatdate.return_value = date
                yield

    def test_extra_kw(self):
        try:
            S3DownloadHandler(Settings(), extra_kw=True)
        except Exception as e:
            self.assertIsInstance(e, (TypeError, NotConfigured))
        else:
            assert False

    def test_request_signing1(self):
        # gets an object from the johnsmith bucket.
        date ='Tue, 27 Mar 2007 19:36:42 +0000'
        req = Request('s3://johnsmith/photos/puppy.jpg', headers={'Date': date})
        with self._mocked_date(date):
            httpreq = self.download_request(req, self.spider)
        self.assertEqual(httpreq.headers['Authorization'], \
                b'AWS 0PN5J17HBGZHT7JJ3X82:xXjDGYUmKxnwqr5KXNPGldn5LbA=')

    def test_request_signing2(self):
        # puts an object into the johnsmith bucket.
        date = 'Tue, 27 Mar 2007 21:15:45 +0000'
        req = Request('s3://johnsmith/photos/puppy.jpg', method='PUT', headers={
            'Content-Type': 'image/jpeg',
            'Date': date,
            'Content-Length': '94328',
            })
        with self._mocked_date(date):
            httpreq = self.download_request(req, self.spider)
        self.assertEqual(httpreq.headers['Authorization'], \
                b'AWS 0PN5J17HBGZHT7JJ3X82:hcicpDDvL9SsO6AkvxqmIWkmOuQ=')

    def test_request_signing3(self):
        # lists the content of the johnsmith bucket.
        date = 'Tue, 27 Mar 2007 19:42:41 +0000'
        req = Request('s3://johnsmith/?prefix=photos&max-keys=50&marker=puppy', \
                method='GET', headers={
                    'User-Agent': 'Mozilla/5.0',
                    'Date': date,
                    })
        with self._mocked_date(date):
            httpreq = self.download_request(req, self.spider)
        self.assertEqual(httpreq.headers['Authorization'], \
                b'AWS 0PN5J17HBGZHT7JJ3X82:jsRt/rhG+Vtp88HrYL706QhE4w4=')

    def test_request_signing4(self):
        # fetches the access control policy sub-resource for the 'johnsmith' bucket.
        date = 'Tue, 27 Mar 2007 19:44:46 +0000'
        req = Request('s3://johnsmith/?acl',
            method='GET', headers={'Date': date})
        with self._mocked_date(date):
            httpreq = self.download_request(req, self.spider)
        self.assertEqual(httpreq.headers['Authorization'], \
                b'AWS 0PN5J17HBGZHT7JJ3X82:thdUi9VAkzhkniLj96JIrOPGi0g=')

    def test_request_signing5(self):
        try: import botocore
        except ImportError: pass
        else:
            raise unittest.SkipTest(
                'botocore does not support overriding date with x-amz-date')
        # deletes an object from the 'johnsmith' bucket using the
        # path-style and Date alternative.
        date = 'Tue, 27 Mar 2007 21:20:27 +0000'
        req = Request('s3://johnsmith/photos/puppy.jpg', \
                method='DELETE', headers={
                    'Date': date,
                    'x-amz-date': 'Tue, 27 Mar 2007 21:20:26 +0000',
                    })
        with self._mocked_date(date):
            httpreq = self.download_request(req, self.spider)
        # botocore does not override Date with x-amz-date
        self.assertEqual(httpreq.headers['Authorization'],
                b'AWS 0PN5J17HBGZHT7JJ3X82:k3nL7gH3+PadhTEVn5Ip83xlYzk=')

    def test_request_signing6(self):
        # uploads an object to a CNAME style virtual hosted bucket with metadata.
        date = 'Tue, 27 Mar 2007 21:06:08 +0000'
        req = Request('s3://static.johnsmith.net:8080/db-backup.dat.gz', \
                method='PUT', headers={
                    'User-Agent': 'curl/7.15.5',
                    'Host': 'static.johnsmith.net:8080',
                    'Date': date,
                    'x-amz-acl': 'public-read',
                    'content-type': 'application/x-download',
                    'Content-MD5': '4gJE4saaMU4BqNR0kLY+lw==',
                    'X-Amz-Meta-ReviewedBy': 'joe@johnsmith.net,jane@johnsmith.net',
                    'X-Amz-Meta-FileChecksum': '0x02661779',
                    'X-Amz-Meta-ChecksumAlgorithm': 'crc32',
                    'Content-Disposition': 'attachment; filename=database.dat',
                    'Content-Encoding': 'gzip',
                    'Content-Length': '5913339',
                    })
        with self._mocked_date(date):
            httpreq = self.download_request(req, self.spider)
        self.assertEqual(httpreq.headers['Authorization'], \
                b'AWS 0PN5J17HBGZHT7JJ3X82:C0FlOtU8Ylb9KDTpZqYkZPX91iI=')

    def test_request_signing7(self):
        # ensure that spaces are quoted properly before signing
        date = 'Tue, 27 Mar 2007 19:42:41 +0000'
        req = Request(
            ("s3://johnsmith/photos/my puppy.jpg"
             "?response-content-disposition=my puppy.jpg"),
            method='GET',
            headers={'Date': date},
            )
        with self._mocked_date(date):
            httpreq = self.download_request(req, self.spider)
        self.assertEqual(
            httpreq.headers['Authorization'],
            b'AWS 0PN5J17HBGZHT7JJ3X82:+CfvG8EZ3YccOrRVMXNaK2eKZmM=')


class ChunkSize3MarshalFifoDiskQueueTest(MarshalFifoDiskQueueTest):
    chunksize = 3

class StdoutFeedStorageTest(unittest.TestCase):

    @defer.inlineCallbacks
    def test_store(self):
        out = BytesIO()
        storage = StdoutFeedStorage('stdout:', _stdout=out)
        file = storage.open(scrapy.Spider("default"))
        file.write(b"content")
        yield storage.store(file)
        self.assertEqual(out.getvalue(), b"content")


class TextResponseTest(BaseResponseTest):

    response_class = TextResponse

    def test_replace(self):
        super(TextResponseTest, self).test_replace()
        r1 = self.response_class("http://www.example.com", body="hello", encoding="cp852")
        r2 = r1.replace(url="http://www.example.com/other")
        r3 = r1.replace(url="http://www.example.com/other", encoding="latin1")

        assert isinstance(r2, self.response_class)
        self.assertEqual(r2.url, "http://www.example.com/other")
        self._assert_response_encoding(r2, "cp852")
        self.assertEqual(r3.url, "http://www.example.com/other")
        self.assertEqual(r3._declared_encoding(), "latin1")

    def test_unicode_url(self):
        # instantiate with unicode url without encoding (should set default encoding)
        resp = self.response_class(u"http://www.example.com/")
        self._assert_response_encoding(resp, self.response_class._DEFAULT_ENCODING)

        # make sure urls are converted to str
        resp = self.response_class(url=u"http://www.example.com/", encoding='utf-8')
        assert isinstance(resp.url, str)

        resp = self.response_class(url=u"http://www.example.com/price/\xa3", encoding='utf-8')
        self.assertEqual(resp.url, to_native_str(b'http://www.example.com/price/\xc2\xa3'))
        resp = self.response_class(url=u"http://www.example.com/price/\xa3", encoding='latin-1')
        self.assertEqual(resp.url, 'http://www.example.com/price/\xa3')
        resp = self.response_class(u"http://www.example.com/price/\xa3", headers={"Content-type": ["text/html; charset=utf-8"]})
        self.assertEqual(resp.url, to_native_str(b'http://www.example.com/price/\xc2\xa3'))
        resp = self.response_class(u"http://www.example.com/price/\xa3", headers={"Content-type": ["text/html; charset=iso-8859-1"]})
        self.assertEqual(resp.url, 'http://www.example.com/price/\xa3')

    def test_unicode_body(self):
        unicode_string = u'\u043a\u0438\u0440\u0438\u043b\u043b\u0438\u0447\u0435\u0441\u043a\u0438\u0439 \u0442\u0435\u043a\u0441\u0442'
        self.assertRaises(TypeError, self.response_class, 'http://www.example.com', body=u'unicode body')

        original_string = unicode_string.encode('cp1251')
        r1 = self.response_class('http://www.example.com', body=original_string, encoding='cp1251')

        # check body_as_unicode
        self.assertTrue(isinstance(r1.body_as_unicode(), six.text_type))
        self.assertEqual(r1.body_as_unicode(), unicode_string)

        # check response.text
        self.assertTrue(isinstance(r1.text, six.text_type))
        self.assertEqual(r1.text, unicode_string)

    def test_encoding(self):
        r1 = self.response_class("http://www.example.com", headers={"Content-type": ["text/html; charset=utf-8"]}, body=b"\xc2\xa3")
        r2 = self.response_class("http://www.example.com", encoding='utf-8', body=u"\xa3")
        r3 = self.response_class("http://www.example.com", headers={"Content-type": ["text/html; charset=iso-8859-1"]}, body=b"\xa3")
        r4 = self.response_class("http://www.example.com", body=b"\xa2\xa3")
        r5 = self.response_class("http://www.example.com", headers={"Content-type": ["text/html; charset=None"]}, body=b"\xc2\xa3")
        r6 = self.response_class("http://www.example.com", headers={"Content-type": ["text/html; charset=gb2312"]}, body=b"\xa8D")
        r7 = self.response_class("http://www.example.com", headers={"Content-type": ["text/html; charset=gbk"]}, body=b"\xa8D")

        self.assertEqual(r1._headers_encoding(), "utf-8")
        self.assertEqual(r2._headers_encoding(), None)
        self.assertEqual(r2._declared_encoding(), 'utf-8')
        self._assert_response_encoding(r2, 'utf-8')
        self.assertEqual(r3._headers_encoding(), "cp1252")
        self.assertEqual(r3._declared_encoding(), "cp1252")
        self.assertEqual(r4._headers_encoding(), None)
        self.assertEqual(r5._headers_encoding(), None)
        self._assert_response_encoding(r5, "utf-8")
        assert r4._body_inferred_encoding() is not None and r4._body_inferred_encoding() != 'ascii'
        self._assert_response_values(r1, 'utf-8', u"\xa3")
        self._assert_response_values(r2, 'utf-8', u"\xa3")
        self._assert_response_values(r3, 'iso-8859-1', u"\xa3")
        self._assert_response_values(r6, 'gb18030', u"\u2015")
        self._assert_response_values(r7, 'gb18030', u"\u2015")

        # TextResponse (and subclasses) must be passed a encoding when instantiating with unicode bodies
        self.assertRaises(TypeError, self.response_class, "http://www.example.com", body=u"\xa3")

    def test_declared_encoding_invalid(self):
        """Check that unknown declared encodings are ignored"""
        r = self.response_class("http://www.example.com",
                                headers={"Content-type": ["text/html; charset=UKNOWN"]},
                                body=b"\xc2\xa3")
        self.assertEqual(r._declared_encoding(), None)
        self._assert_response_values(r, 'utf-8', u"\xa3")

    def test_utf16(self):
        """Test utf-16 because UnicodeDammit is known to have problems with"""
        r = self.response_class("http://www.example.com",
                                body=b'\xff\xfeh\x00i\x00',
                                encoding='utf-16')
        self._assert_response_values(r, 'utf-16', u"hi")

    def test_invalid_utf8_encoded_body_with_valid_utf8_BOM(self):
        r6 = self.response_class("http://www.example.com",
                                 headers={"Content-type": ["text/html; charset=utf-8"]},
                                 body=b"\xef\xbb\xbfWORD\xe3\xab")
        self.assertEqual(r6.encoding, 'utf-8')
        self.assertEqual(r6.text, u'WORD\ufffd\ufffd')

    def test_bom_is_removed_from_body(self):
        # Inferring encoding from body also cache decoded body as sideeffect,
        # this test tries to ensure that calling response.encoding and
        # response.text in indistint order doesn't affect final
        # values for encoding and decoded body.
        url = 'http://example.com'
        body = b"\xef\xbb\xbfWORD"
        headers = {"Content-type": ["text/html; charset=utf-8"]}

        # Test response without content-type and BOM encoding
        response = self.response_class(url, body=body)
        self.assertEqual(response.encoding, 'utf-8')
        self.assertEqual(response.text, u'WORD')
        response = self.response_class(url, body=body)
        self.assertEqual(response.text, u'WORD')
        self.assertEqual(response.encoding, 'utf-8')

        # Body caching sideeffect isn't triggered when encoding is declared in
        # content-type header but BOM still need to be removed from decoded
        # body
        response = self.response_class(url, headers=headers, body=body)
        self.assertEqual(response.encoding, 'utf-8')
        self.assertEqual(response.text, u'WORD')
        response = self.response_class(url, headers=headers, body=body)
        self.assertEqual(response.text, u'WORD')
        self.assertEqual(response.encoding, 'utf-8')

    def test_replace_wrong_encoding(self):
        """Test invalid chars are replaced properly"""
        r = self.response_class("http://www.example.com", encoding='utf-8', body=b'PREFIX\xe3\xabSUFFIX')
        # XXX: Policy for replacing invalid chars may suffer minor variations
        # but it should always contain the unicode replacement char (u'\ufffd')
        assert u'\ufffd' in r.text, repr(r.text)
        assert u'PREFIX' in r.text, repr(r.text)
        assert u'SUFFIX' in r.text, repr(r.text)

        # Do not destroy html tags due to encoding bugs
        r = self.response_class("http://example.com", encoding='utf-8', \
                body=b'\xf0<span>value</span>')
        assert u'<span>value</span>' in r.text, repr(r.text)

        # FIXME: This test should pass once we stop using BeautifulSoup's UnicodeDammit in TextResponse
        #r = self.response_class("http://www.example.com", body=b'PREFIX\xe3\xabSUFFIX')
        #assert u'\ufffd' in r.text, repr(r.text)

    def test_selector(self):
        body = b"<html><head><title>Some page</title><body></body></html>"
        response = self.response_class("http://www.example.com", body=body)

        self.assertIsInstance(response.selector, Selector)
        self.assertEqual(response.selector.type, 'html')
        self.assertIs(response.selector, response.selector)  # property is cached
        self.assertIs(response.selector.response, response)

        self.assertEqual(
            response.selector.xpath("//title/text()").extract(),
            [u'Some page']
        )
        self.assertEqual(
            response.selector.css("title::text").extract(),
            [u'Some page']
        )
        self.assertEqual(
            response.selector.re("Some (.*)</title>"),
            [u'page']
        )

    def test_selector_shortcuts(self):
        body = b"<html><head><title>Some page</title><body></body></html>"
        response = self.response_class("http://www.example.com", body=body)

        self.assertEqual(
            response.xpath("//title/text()").extract(),
            response.selector.xpath("//title/text()").extract(),
        )
        self.assertEqual(
            response.css("title::text").extract(),
            response.selector.css("title::text").extract(),
        )

    def test_selector_shortcuts_kwargs(self):
        body = b"<html><head><title>Some page</title><body><p class=\"content\">A nice paragraph.</p></body></html>"
        response = self.response_class("http://www.example.com", body=body)

        self.assertEqual(
            response.xpath("normalize-space(//p[@class=$pclass])", pclass="content").extract(),
            response.xpath("normalize-space(//p[@class=\"content\"])").extract(),
        )
        self.assertEqual(
            response.xpath("//title[count(following::p[@class=$pclass])=$pcount]/text()",
                pclass="content", pcount=1).extract(),
            response.xpath("//title[count(following::p[@class=\"content\"])=1]/text()").extract(),
        )

    def test_urljoin_with_base_url(self):
        """Test urljoin shortcut which also evaluates base-url through get_base_url()."""
        body = b'<html><body><base href="https://example.net"></body></html>'
        joined = self.response_class('http://www.example.com', body=body).urljoin('/test')
        absolute = 'https://example.net/test'
        self.assertEqual(joined, absolute)

        body = b'<html><body><base href="/elsewhere"></body></html>'
        joined = self.response_class('http://www.example.com', body=body).urljoin('test')
        absolute = 'http://www.example.com/test'
        self.assertEqual(joined, absolute)

        body = b'<html><body><base href="/elsewhere/"></body></html>'
        joined = self.response_class('http://www.example.com', body=body).urljoin('test')
        absolute = 'http://www.example.com/elsewhere/test'
        self.assertEqual(joined, absolute)

    def test_follow_selector(self):
        resp = self._links_response()
        urls = [
            'http://example.com/sample2.html',
            'http://example.com/sample3.html',
            'http://example.com/sample3.html',
            'http://example.com/sample3.html#foo',
            'http://www.google.com/something',
            'http://example.com/innertag.html'
        ]

        # select <a> elements
        for sellist in [resp.css('a'), resp.xpath('//a')]:
            for sel, url in zip(sellist, urls):
                self._assert_followed_url(sel, url, response=resp)

        # select <link> elements
        self._assert_followed_url(
            Selector(text='<link href="foo"></link>').css('link')[0],
            'http://example.com/foo',
            response=resp
        )

        # href attributes should work
        for sellist in [resp.css('a::attr(href)'), resp.xpath('//a/@href')]:
            for sel, url in zip(sellist, urls):
                self._assert_followed_url(sel, url, response=resp)

        # non-a elements are not supported
        self.assertRaises(ValueError, resp.follow, resp.css('div')[0])

    def test_follow_selector_list(self):
        resp = self._links_response()
        self.assertRaisesRegexp(ValueError, 'SelectorList',
                                resp.follow, resp.css('a'))

    def test_follow_selector_invalid(self):
        resp = self._links_response()
        self.assertRaisesRegexp(ValueError, 'Unsupported',
                                resp.follow, resp.xpath('count(//div)')[0])

    def test_follow_selector_attribute(self):
        resp = self._links_response()
        for src in resp.css('img::attr(src)'):
            self._assert_followed_url(src, 'http://example.com/sample2.jpg')

    def test_follow_selector_no_href(self):
        resp = self.response_class(
            url='http://example.com',
            body=b'<html><body><a name=123>click me</a></body></html>',
        )
        self.assertRaisesRegexp(ValueError, 'no href',
                                resp.follow, resp.css('a')[0])

    def test_follow_whitespace_selector(self):
        resp = self.response_class(
            'http://example.com',
            body=b'''<html><body><a href=" foo\n">click me</a></body></html>'''
        )
        self._assert_followed_url(resp.css('a')[0],
                                 'http://example.com/foo',
                                  response=resp)
        self._assert_followed_url(resp.css('a::attr(href)')[0],
                                 'http://example.com/foo',
                                  response=resp)

    def test_follow_encoding(self):
        resp1 = self.response_class(
            'http://example.com',
            encoding='utf8',
            body=u'<html><body><a href="foo?привет">click me</a></body></html>'.encode('utf8')
        )
        req = self._assert_followed_url(
            resp1.css('a')[0],
            'http://example.com/foo?%D0%BF%D1%80%D0%B8%D0%B2%D0%B5%D1%82',
            response=resp1,
        )
        self.assertEqual(req.encoding, 'utf8')

        resp2 = self.response_class(
            'http://example.com',
            encoding='cp1251',
            body=u'<html><body><a href="foo?привет">click me</a></body></html>'.encode('cp1251')
        )
        req = self._assert_followed_url(
            resp2.css('a')[0],
            'http://example.com/foo?%EF%F0%E8%E2%E5%F2',
            response=resp2,
        )
        self.assertEqual(req.encoding, 'cp1251')


class TestPolicyHeaderPredecence004(MixinNoReferrerWhenDowngrade, TestRefererMiddleware):
    """
    The empty string means "no-referrer-when-downgrade"
    """
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.OriginWhenCrossOriginPolicy'}
    resp_headers = {'Referrer-Policy': ''}


class CrawlerRun(object):
    """A class to run the crawler and keep track of events occurred"""

    def __init__(self, spider_class):
        self.spider = None
        self.respplug = []
        self.reqplug = []
        self.reqdropped = []
        self.itemresp = []
        self.signals_catched = {}
        self.spider_class = spider_class

    def run(self):
        self.port = start_test_site()
        self.portno = self.port.getHost().port

        start_urls = [self.geturl("/"), self.geturl("/redirect"),
                      self.geturl("/redirect")]  # a duplicate

        for name, signal in vars(signals).items():
            if not name.startswith('_'):
                dispatcher.connect(self.record_signal, signal)

        self.crawler = get_crawler(self.spider_class)
        self.crawler.signals.connect(self.item_scraped, signals.item_scraped)
        self.crawler.signals.connect(self.request_scheduled, signals.request_scheduled)
        self.crawler.signals.connect(self.request_dropped, signals.request_dropped)
        self.crawler.signals.connect(self.response_downloaded, signals.response_downloaded)
        self.crawler.crawl(start_urls=start_urls)
        self.spider = self.crawler.spider

        self.deferred = defer.Deferred()
        dispatcher.connect(self.stop, signals.engine_stopped)
        return self.deferred

    def stop(self):
        self.port.stopListening()
        for name, signal in vars(signals).items():
            if not name.startswith('_'):
                disconnect_all(signal)
        self.deferred.callback(None)

    def geturl(self, path):
        return "http://localhost:%s%s" % (self.portno, path)

    def getpath(self, url):
        u = urlparse(url)
        return u.path

    def item_scraped(self, item, spider, response):
        self.itemresp.append((item, response))

    def request_scheduled(self, request, spider):
        self.reqplug.append((request, spider))

    def request_dropped(self, request, spider):
        self.reqdropped.append((request, spider))

    def response_downloaded(self, response, spider):
        self.respplug.append((response, spider))

    def record_signal(self, *args, **kwargs):
        """Record a signal and its parameters"""
        signalargs = kwargs.copy()
        sig = signalargs.pop('signal')
        signalargs.pop('sender', None)
        self.signals_catched[sig] = signalargs


class TestSettingsSameOrigin(MixinSameOrigin, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.SameOriginPolicy'}

