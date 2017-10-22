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


class TopLevelFormatterTest(unittest.TestCase):

    def setUp(self):
        self.handler = LogCapture()
        self.handler.addFilter(TopLevelFormatter(['test']))

    def test_top_level_logger(self):
        logger = logging.getLogger('test')
        with self.handler as l:
            logger.warning('test log msg')

        l.check(('test', 'WARNING', 'test log msg'))

    def test_children_logger(self):
        logger = logging.getLogger('test.test1')
        with self.handler as l:
            logger.warning('test log msg')

        l.check(('test', 'WARNING', 'test log msg'))

    def test_overlapping_name_logger(self):
        logger = logging.getLogger('test2')
        with self.handler as l:
            logger.warning('test log msg')

        l.check(('test2', 'WARNING', 'test log msg'))

    def test_different_name_logger(self):
        logger = logging.getLogger('different')
        with self.handler as l:
            logger.warning('test log msg')

        l.check(('different', 'WARNING', 'test log msg'))


class ChunkSize4MarshalFifoDiskQueueTest(MarshalFifoDiskQueueTest):
    chunksize = 4


class SitemapTest(unittest.TestCase):

    def test_sitemap(self):
        s = Sitemap(b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.google.com/schemas/sitemap/0.84">
  <url>
    <loc>http://www.example.com/</loc>
    <lastmod>2009-08-16</lastmod>
    <changefreq>daily</changefreq>
    <priority>1</priority>
  </url>
  <url>
    <loc>http://www.example.com/Special-Offers.html</loc>
    <lastmod>2009-08-16</lastmod>
    <changefreq>weekly</changefreq>
    <priority>0.8</priority>
  </url>
</urlset>""")
        assert s.type == 'urlset'
        self.assertEqual(list(s),
            [{'priority': '1', 'loc': 'http://www.example.com/', 'lastmod': '2009-08-16', 'changefreq': 'daily'}, {'priority': '0.8', 'loc': 'http://www.example.com/Special-Offers.html', 'lastmod': '2009-08-16', 'changefreq': 'weekly'}])

    def test_sitemap_index(self):
        s = Sitemap(b"""<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
   <sitemap>
      <loc>http://www.example.com/sitemap1.xml.gz</loc>
      <lastmod>2004-10-01T18:23:17+00:00</lastmod>
   </sitemap>
   <sitemap>
      <loc>http://www.example.com/sitemap2.xml.gz</loc>
      <lastmod>2005-01-01</lastmod>
   </sitemap>
</sitemapindex>""")
        assert s.type == 'sitemapindex'
        self.assertEqual(list(s), [{'loc': 'http://www.example.com/sitemap1.xml.gz', 'lastmod': '2004-10-01T18:23:17+00:00'}, {'loc': 'http://www.example.com/sitemap2.xml.gz', 'lastmod': '2005-01-01'}])

    def test_sitemap_strip(self):
        """Assert we can deal with trailing spaces inside <loc> tags - we've
        seen those
        """
        s = Sitemap(b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.google.com/schemas/sitemap/0.84">
  <url>
    <loc> http://www.example.com/</loc>
    <lastmod>2009-08-16</lastmod>
    <changefreq>daily</changefreq>
    <priority>1</priority>
  </url>
  <url>
    <loc> http://www.example.com/2</loc>
    <lastmod />
  </url>
</urlset>
""")
        self.assertEqual(list(s),
            [{'priority': '1', 'loc': 'http://www.example.com/', 'lastmod': '2009-08-16', 'changefreq': 'daily'},
             {'loc': 'http://www.example.com/2', 'lastmod': ''},
            ])

    def test_sitemap_wrong_ns(self):
        """We have seen sitemaps with wrongs ns. Presumably, Google still works
        with these, though is not 100% confirmed"""
        s = Sitemap(b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.google.com/schemas/sitemap/0.84">
  <url xmlns="">
    <loc> http://www.example.com/</loc>
    <lastmod>2009-08-16</lastmod>
    <changefreq>daily</changefreq>
    <priority>1</priority>
  </url>
  <url xmlns="">
    <loc> http://www.example.com/2</loc>
    <lastmod />
  </url>
</urlset>
""")
        self.assertEqual(list(s),
            [{'priority': '1', 'loc': 'http://www.example.com/', 'lastmod': '2009-08-16', 'changefreq': 'daily'},
             {'loc': 'http://www.example.com/2', 'lastmod': ''},
            ])

    def test_sitemap_wrong_ns2(self):
        """We have seen sitemaps with wrongs ns. Presumably, Google still works
        with these, though is not 100% confirmed"""
        s = Sitemap(b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset>
  <url xmlns="">
    <loc> http://www.example.com/</loc>
    <lastmod>2009-08-16</lastmod>
    <changefreq>daily</changefreq>
    <priority>1</priority>
  </url>
  <url xmlns="">
    <loc> http://www.example.com/2</loc>
    <lastmod />
  </url>
</urlset>
""")
        assert s.type == 'urlset'
        self.assertEqual(list(s),
            [{'priority': '1', 'loc': 'http://www.example.com/', 'lastmod': '2009-08-16', 'changefreq': 'daily'},
             {'loc': 'http://www.example.com/2', 'lastmod': ''},
            ])

    def test_sitemap_urls_from_robots(self):
        robots = """User-agent: *
Disallow: /aff/
Disallow: /wl/

# Search and shopping refining
Disallow: /s*/*facet
Disallow: /s*/*tags

# Sitemap files
Sitemap: http://example.com/sitemap.xml
Sitemap: http://example.com/sitemap-product-index.xml
Sitemap: HTTP://example.com/sitemap-uppercase.xml
Sitemap: /sitemap-relative-url.xml

# Forums
Disallow: /forum/search/
Disallow: /forum/active/
"""
        self.assertEqual(list(sitemap_urls_from_robots(robots, base_url='http://example.com')),
                         ['http://example.com/sitemap.xml',
                          'http://example.com/sitemap-product-index.xml',
                          'http://example.com/sitemap-uppercase.xml',
                          'http://example.com/sitemap-relative-url.xml'])

    def test_sitemap_blanklines(self):
        """Assert we can deal with starting blank lines before <xml> tag"""
        s = Sitemap(b"""\

<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">

<!-- cache: cached = yes name = sitemap_jspCache key = sitemap -->
<sitemap>
<loc>http://www.example.com/sitemap1.xml</loc>
<lastmod>2013-07-15</lastmod>
</sitemap>

<sitemap>
<loc>http://www.example.com/sitemap2.xml</loc>
<lastmod>2013-07-15</lastmod>
</sitemap>

<sitemap>
<loc>http://www.example.com/sitemap3.xml</loc>
<lastmod>2013-07-15</lastmod>
</sitemap>

<!-- end cache -->
</sitemapindex>
""")
        self.assertEqual(list(s), [
            {'lastmod': '2013-07-15', 'loc': 'http://www.example.com/sitemap1.xml'},
            {'lastmod': '2013-07-15', 'loc': 'http://www.example.com/sitemap2.xml'},
            {'lastmod': '2013-07-15', 'loc': 'http://www.example.com/sitemap3.xml'},
        ])

    def test_comment(self):
        s = Sitemap(b"""<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:xhtml="http://www.w3.org/1999/xhtml">
        <url>
            <loc>http://www.example.com/</loc>
            <!-- this is a comment on which the parser might raise an exception if implemented incorrectly -->
        </url>
    </urlset>""")

        self.assertEqual(list(s), [
            {'loc': 'http://www.example.com/'}
        ])

    def test_alternate(self):
        s = Sitemap(b"""<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:xhtml="http://www.w3.org/1999/xhtml">
        <url>
            <loc>http://www.example.com/english/</loc>
            <xhtml:link rel="alternate" hreflang="de"
                href="http://www.example.com/deutsch/"/>
            <xhtml:link rel="alternate" hreflang="de-ch"
                href="http://www.example.com/schweiz-deutsch/"/>
            <xhtml:link rel="alternate" hreflang="en"
                href="http://www.example.com/english/"/>
            <xhtml:link rel="alternate" hreflang="en"/><!-- wrong tag without href -->
        </url>
    </urlset>""")

        self.assertEqual(list(s), [
            {'loc': 'http://www.example.com/english/',
             'alternate': ['http://www.example.com/deutsch/', 'http://www.example.com/schweiz-deutsch/', 'http://www.example.com/english/']
            }
        ])

    def test_xml_entity_expansion(self):
        s = Sitemap(b"""<?xml version="1.0" encoding="utf-8"?>
          <!DOCTYPE foo [
          <!ELEMENT foo ANY >
          <!ENTITY xxe SYSTEM "file:///etc/passwd" >
          ]>
          <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
              <loc>http://127.0.0.1:8000/&xxe;</loc>
            </url>
          </urlset>
        """)

        self.assertEqual(list(s), [{'loc': 'http://127.0.0.1:8000/'}])


if __name__ == '__main__':
    unittest.main()

class TestReferrerOnRedirect(TestRefererMiddleware):

    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.UnsafeUrlPolicy'}
    scenarii = [
        (   'http://scrapytest.org/1',      # parent
            'http://scrapytest.org/2',      # target
            (
                # redirections: code, URL
                (301, 'http://scrapytest.org/3'),
                (301, 'http://scrapytest.org/4'),
            ),
            b'http://scrapytest.org/1', # expected initial referer
            b'http://scrapytest.org/1', # expected referer for the redirection request
        ),
        (   'https://scrapytest.org/1',
            'https://scrapytest.org/2',
            (
                # redirecting to non-secure URL
                (301, 'http://scrapytest.org/3'),
            ),
            b'https://scrapytest.org/1',
            b'https://scrapytest.org/1',
        ),
        (   'https://scrapytest.org/1',
            'https://scrapytest.com/2',
            (
                # redirecting to non-secure URL: different origin
                (301, 'http://scrapytest.com/3'),
            ),
            b'https://scrapytest.org/1',
            b'https://scrapytest.org/1',
        ),
    ]

    def setUp(self):
        self.spider = Spider('foo')
        settings = Settings(self.settings)
        self.referrermw = RefererMiddleware(settings)
        self.redirectmw = RedirectMiddleware(settings)

    def test(self):

        for parent, target, redirections, init_referrer, final_referrer in self.scenarii:
            response = self.get_response(parent)
            request = self.get_request(target)

            out = list(self.referrermw.process_spider_output(response, [request], self.spider))
            self.assertEqual(out[0].headers.get('Referer'), init_referrer)

            for status, url in redirections:
                response = Response(request.url, headers={'Location': url}, status=status)
                request = self.redirectmw.process_response(request, response, self.spider)
                self.referrermw.request_scheduled(request, self.spider)

            assert isinstance(request, Request)
            self.assertEqual(request.headers.get('Referer'), final_referrer)


class CustomRequest(Request):
    pass

class TestItem(Item):
    name = Field()

def _test_procesor(x):
    return x + x

class UtilsRenderTemplateFileTestCase(unittest.TestCase):

    def setUp(self):
        self.tmp_path = mkdtemp()

    def tearDown(self):
        rmtree(self.tmp_path)

    def test_simple_render(self):

        context = dict(project_name='proj', name='spi', classname='TheSpider')
        template = u'from ${project_name}.spiders.${name} import ${classname}'
        rendered = u'from proj.spiders.spi import TheSpider'

        template_path = os.path.join(self.tmp_path, 'templ.py.tmpl')
        render_path = os.path.join(self.tmp_path, 'templ.py')

        with open(template_path, 'wb') as tmpl_file:
            tmpl_file.write(template.encode('utf8'))
        assert os.path.isfile(template_path)  # Failure of test itself

        render_templatefile(template_path, **context)

        self.assertFalse(os.path.exists(template_path))
        with open(render_path, 'rb') as result:
            self.assertEqual(result.read().decode('utf8'), rendered)

        os.remove(render_path)
        assert not os.path.exists(render_path)  # Failure of test iself

if '__main__' == __name__:
    unittest.main()

class TestRequestMetaPredecence001(MixinUnsafeUrl, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.SameOriginPolicy'}
    req_meta = {'referrer_policy': POLICY_UNSAFE_URL}


class DeprecatedPydispatchTest(unittest.TestCase):
    def test_import_xlib_pydispatch_show_warning(self):
        with warnings.catch_warnings(record=True) as w:
            from scrapy.xlib import pydispatch
            reload_module(pydispatch)
        self.assertIn('Importing from scrapy.xlib.pydispatch is deprecated',
                      str(w[0].message))

class HtmlResponseTest(TextResponseTest):

    response_class = HtmlResponse

    def test_html_encoding(self):

        body = b"""<html><head><title>Some page</title><meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">
        </head><body>Price: \xa3100</body></html>'
        """
        r1 = self.response_class("http://www.example.com", body=body)
        self._assert_response_values(r1, 'iso-8859-1', body)

        body = b"""<?xml version="1.0" encoding="iso-8859-1"?>
        <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
        Price: \xa3100
        """
        r2 = self.response_class("http://www.example.com", body=body)
        self._assert_response_values(r2, 'iso-8859-1', body)

        # for conflicting declarations headers must take precedence
        body = b"""<html><head><title>Some page</title><meta http-equiv="Content-Type" content="text/html; charset=utf-8">
        </head><body>Price: \xa3100</body></html>'
        """
        r3 = self.response_class("http://www.example.com", headers={"Content-type": ["text/html; charset=iso-8859-1"]}, body=body)
        self._assert_response_values(r3, 'iso-8859-1', body)

        # make sure replace() preserves the encoding of the original response
        body = b"New body \xa3"
        r4 = r3.replace(body=body)
        self._assert_response_values(r4, 'iso-8859-1', body)

    def test_html5_meta_charset(self):
        body = b"""<html><head><meta charset="gb2312" /><title>Some page</title><body>bla bla</body>"""
        r1 = self.response_class("http://www.example.com", body=body)
        self._assert_response_values(r1, 'gb2312', body)


class TestPolicyHeaderPredecence001(MixinUnsafeUrl, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.SameOriginPolicy'}
    resp_headers = {'Referrer-Policy': POLICY_UNSAFE_URL.upper()}

class GunzipTest(unittest.TestCase):

    def test_gunzip_basic(self):
        with open(join(SAMPLEDIR, 'feed-sample1.xml.gz'), 'rb') as f:
            text = gunzip(f.read())
            self.assertEqual(len(text), 9950)

    def test_gunzip_truncated(self):
        with open(join(SAMPLEDIR, 'truncated-crc-error.gz'), 'rb') as f:
            text = gunzip(f.read())
            assert text.endswith(b'</html')

    def test_gunzip_no_gzip_file_raises(self):
        with open(join(SAMPLEDIR, 'feed-sample1.xml'), 'rb') as f:
            self.assertRaises(IOError, gunzip, f.read())

    def test_gunzip_truncated_short(self):
        with open(join(SAMPLEDIR, 'truncated-crc-error-short.gz'), 'rb') as f:
            text = gunzip(f.read())
            assert text.endswith(b'</html>')

    def test_is_x_gzipped_right(self):
        hdrs = Headers({"Content-Type": "application/x-gzip"})
        r1 = Response("http://www.example.com", headers=hdrs)
        self.assertTrue(is_gzipped(r1))

    def test_is_gzipped_right(self):
        hdrs = Headers({"Content-Type": "application/gzip"})
        r1 = Response("http://www.example.com", headers=hdrs)
        self.assertTrue(is_gzipped(r1))

    def test_is_gzipped_not_quite(self):
        hdrs = Headers({"Content-Type": "application/gzippppp"})
        r1 = Response("http://www.example.com", headers=hdrs)
        self.assertFalse(is_gzipped(r1))

    def test_is_gzipped_case_insensitive(self):
        hdrs = Headers({"Content-Type": "Application/X-Gzip"})
        r1 = Response("http://www.example.com", headers=hdrs)
        self.assertTrue(is_gzipped(r1))

        hdrs = Headers({"Content-Type": "application/X-GZIP ; charset=utf-8"})
        r1 = Response("http://www.example.com", headers=hdrs)
        self.assertTrue(is_gzipped(r1))

    def test_is_gzipped_empty(self):
        r1 = Response("http://www.example.com")
        self.assertFalse(is_gzipped(r1))

    def test_is_gzipped_wrong(self):
        hdrs = Headers({"Content-Type": "application/javascript"})
        r1 = Response("http://www.example.com", headers=hdrs)
        self.assertFalse(is_gzipped(r1))

    def test_is_gzipped_with_charset(self):
        hdrs = Headers({"Content-Type": "application/x-gzip;charset=utf-8"})
        r1 = Response("http://www.example.com", headers=hdrs)
        self.assertTrue(is_gzipped(r1))

    def test_gunzip_illegal_eof(self):
        with open(join(SAMPLEDIR, 'unexpected-eof.gz'), 'rb') as f:
            text = html_to_unicode('charset=cp1252', gunzip(f.read()))[1]
            with open(join(SAMPLEDIR, 'unexpected-eof-output.txt'), 'rb') as o:
                expected_text = o.read().decode("utf-8")
                self.assertEqual(len(text), len(expected_text))
                self.assertEqual(text, expected_text)

class MixinDefault(object):
    """
    Based on https://www.w3.org/TR/referrer-policy/#referrer-policy-no-referrer-when-downgrade

    with some additional filtering of s3://
    """
    scenarii = [
        ('https://example.com/',    'https://scrapy.org/',  b'https://example.com/'),
        ('http://example.com/',     'http://scrapy.org/',   b'http://example.com/'),
        ('http://example.com/',     'https://scrapy.org/',  b'http://example.com/'),
        ('https://example.com/',    'http://scrapy.org/',   None),

        # no credentials leak
        ('http://user:password@example.com/',  'https://scrapy.org/', b'http://example.com/'),

        # no referrer leak for local schemes
        ('file:///home/path/to/somefile.html',  'https://scrapy.org/', None),
        ('file:///home/path/to/somefile.html',  'http://scrapy.org/',  None),

        # no referrer leak for s3 origins
        ('s3://mybucket/path/to/data.csv',  'https://scrapy.org/', None),
        ('s3://mybucket/path/to/data.csv',  'http://scrapy.org/',  None),
    ]


class Bar(trackref.object_ref):
    pass


class JsonLinesItemExporterTest(BaseItemExporterTest):

    _expected_nested = {'name': u'Jesus', 'age': {'name': 'Maria', 'age': {'name': 'Joseph', 'age': '22'}}}

    def _get_exporter(self, **kwargs):
        return JsonLinesItemExporter(self.output, **kwargs)

    def _check_output(self):
        exported = json.loads(to_unicode(self.output.getvalue().strip()))
        self.assertEqual(exported, dict(self.i))

    def test_nested_item(self):
        i1 = TestItem(name=u'Joseph', age='22')
        i2 = dict(name=u'Maria', age=i1)
        i3 = TestItem(name=u'Jesus', age=i2)
        self.ie.start_exporting()
        self.ie.export_item(i3)
        self.ie.finish_exporting()
        exported = json.loads(to_unicode(self.output.getvalue()))
        self.assertEqual(exported, self._expected_nested)

    def test_extra_keywords(self):
        self.ie = self._get_exporter(sort_keys=True)
        self.test_export_item()
        self._check_output()
        self.assertRaises(TypeError, self._get_exporter, foo_unknown_keyword_bar=True)

    def test_nonstring_types_item(self):
        item = self._get_nonstring_types_item()
        self.ie.start_exporting()
        self.ie.export_item(item)
        self.ie.finish_exporting()
        exported = json.loads(to_unicode(self.output.getvalue()))
        item['time'] = str(item['time'])
        self.assertEqual(exported, item)


class XmlResponseTest(TextResponseTest):

    response_class = XmlResponse

    def test_xml_encoding(self):
        body = b"<xml></xml>"
        r1 = self.response_class("http://www.example.com", body=body)
        self._assert_response_values(r1, self.response_class._DEFAULT_ENCODING, body)

        body = b"""<?xml version="1.0" encoding="iso-8859-1"?><xml></xml>"""
        r2 = self.response_class("http://www.example.com", body=body)
        self._assert_response_values(r2, 'iso-8859-1', body)

        # make sure replace() preserves the explicit encoding passed in the constructor
        body = b"""<?xml version="1.0" encoding="iso-8859-1"?><xml></xml>"""
        r3 = self.response_class("http://www.example.com", body=body, encoding='utf-8')
        body2 = b"New body"
        r4 = r3.replace(body=body2)
        self._assert_response_values(r4, 'utf-8', body2)

    def test_replace_encoding(self):
        # make sure replace() keeps the previous encoding unless overridden explicitly
        body = b"""<?xml version="1.0" encoding="iso-8859-1"?><xml></xml>"""
        body2 = b"""<?xml version="1.0" encoding="utf-8"?><xml></xml>"""
        r5 = self.response_class("http://www.example.com", body=body)
        r6 = r5.replace(body=body2)
        r7 = r5.replace(body=body2, encoding='utf-8')
        self._assert_response_values(r5, 'iso-8859-1', body)
        self._assert_response_values(r6, 'iso-8859-1', body2)
        self._assert_response_values(r7, 'utf-8', body2)

    def test_selector(self):
        body = b'<?xml version="1.0" encoding="utf-8"?><xml><elem>value</elem></xml>'
        response = self.response_class("http://www.example.com", body=body)

        self.assertIsInstance(response.selector, Selector)
        self.assertEqual(response.selector.type, 'xml')
        self.assertIs(response.selector, response.selector)  # property is cached
        self.assertIs(response.selector.response, response)

        self.assertEqual(
            response.selector.xpath("//elem/text()").extract(),
            [u'value']
        )

    def test_selector_shortcuts(self):
        body = b'<?xml version="1.0" encoding="utf-8"?><xml><elem>value</elem></xml>'
        response = self.response_class("http://www.example.com", body=body)

        self.assertEqual(
            response.xpath("//elem/text()").extract(),
            response.selector.xpath("//elem/text()").extract(),
        )

    def test_selector_shortcuts_kwargs(self):
        body = b'''<?xml version="1.0" encoding="utf-8"?>
        <xml xmlns:somens="http://scrapy.org">
        <somens:elem>value</somens:elem>
        </xml>'''
        response = self.response_class("http://www.example.com", body=body)

        self.assertEqual(
            response.xpath("//s:elem/text()", namespaces={'s': 'http://scrapy.org'}).extract(),
            response.selector.xpath("//s:elem/text()", namespaces={'s': 'http://scrapy.org'}).extract(),
        )

        response.selector.register_namespace('s2', 'http://scrapy.org')
        self.assertEqual(
            response.xpath("//s1:elem/text()", namespaces={'s1': 'http://scrapy.org'}).extract(),
            response.selector.xpath("//s2:elem/text()").extract(),
        )

class TestRequestMetaPredecence003(MixinUnsafeUrl, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.OriginWhenCrossOriginPolicy'}
    req_meta = {'referrer_policy': POLICY_UNSAFE_URL}


class M1(object):

    def open_spider(self, spider):
        pass

    def close_spider(self, spider):
        pass

    def process(self, response, request, spider):
        pass

class MixinStrictOrigin(object):
    scenarii = [
        # TLS or non-TLS to TLS or non-TLS: referrer origin is sent but not for downgrades
        ('https://example.com/page.html',   'https://example.com/not-page.html',    b'https://example.com/'),
        ('https://example.com/page.html',   'https://scrapy.org',                   b'https://example.com/'),
        ('http://example.com/page.html',    'http://scrapy.org',                    b'http://example.com/'),

        # downgrade: send nothing
        ('https://example.com/page.html',   'http://scrapy.org',                    None),

        # upgrade: send origin
        ('http://example.com/page.html',    'https://scrapy.org',                   b'http://example.com/'),

        # test for user/password stripping
        ('https://user:password@example.com/page.html', 'https://scrapy.org',       b'https://example.com/'),
        ('https://user:password@example.com/page.html', 'http://scrapy.org',        None),
    ]


class DeprecatedImagesPipeline(ImagesPipeline):
    def file_key(self, url):
        return self.image_key(url)

    def image_key(self, url):
        image_guid = hashlib.sha1(to_bytes(url)).hexdigest()
        return 'empty/%s.jpg' % (image_guid)

    def thumb_key(self, url, thumb_id):
        thumb_guid = hashlib.sha1(to_bytes(url)).hexdigest()
        return 'thumbsup/%s/%s.jpg' % (thumb_id, thumb_guid)


class SitemapTest(unittest.TestCase):

    def test_sitemap(self):
        s = Sitemap(b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.google.com/schemas/sitemap/0.84">
  <url>
    <loc>http://www.example.com/</loc>
    <lastmod>2009-08-16</lastmod>
    <changefreq>daily</changefreq>
    <priority>1</priority>
  </url>
  <url>
    <loc>http://www.example.com/Special-Offers.html</loc>
    <lastmod>2009-08-16</lastmod>
    <changefreq>weekly</changefreq>
    <priority>0.8</priority>
  </url>
</urlset>""")
        assert s.type == 'urlset'
        self.assertEqual(list(s),
            [{'priority': '1', 'loc': 'http://www.example.com/', 'lastmod': '2009-08-16', 'changefreq': 'daily'}, {'priority': '0.8', 'loc': 'http://www.example.com/Special-Offers.html', 'lastmod': '2009-08-16', 'changefreq': 'weekly'}])

    def test_sitemap_index(self):
        s = Sitemap(b"""<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
   <sitemap>
      <loc>http://www.example.com/sitemap1.xml.gz</loc>
      <lastmod>2004-10-01T18:23:17+00:00</lastmod>
   </sitemap>
   <sitemap>
      <loc>http://www.example.com/sitemap2.xml.gz</loc>
      <lastmod>2005-01-01</lastmod>
   </sitemap>
</sitemapindex>""")
        assert s.type == 'sitemapindex'
        self.assertEqual(list(s), [{'loc': 'http://www.example.com/sitemap1.xml.gz', 'lastmod': '2004-10-01T18:23:17+00:00'}, {'loc': 'http://www.example.com/sitemap2.xml.gz', 'lastmod': '2005-01-01'}])

    def test_sitemap_strip(self):
        """Assert we can deal with trailing spaces inside <loc> tags - we've
        seen those
        """
        s = Sitemap(b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.google.com/schemas/sitemap/0.84">
  <url>
    <loc> http://www.example.com/</loc>
    <lastmod>2009-08-16</lastmod>
    <changefreq>daily</changefreq>
    <priority>1</priority>
  </url>
  <url>
    <loc> http://www.example.com/2</loc>
    <lastmod />
  </url>
</urlset>
""")
        self.assertEqual(list(s),
            [{'priority': '1', 'loc': 'http://www.example.com/', 'lastmod': '2009-08-16', 'changefreq': 'daily'},
             {'loc': 'http://www.example.com/2', 'lastmod': ''},
            ])

    def test_sitemap_wrong_ns(self):
        """We have seen sitemaps with wrongs ns. Presumably, Google still works
        with these, though is not 100% confirmed"""
        s = Sitemap(b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.google.com/schemas/sitemap/0.84">
  <url xmlns="">
    <loc> http://www.example.com/</loc>
    <lastmod>2009-08-16</lastmod>
    <changefreq>daily</changefreq>
    <priority>1</priority>
  </url>
  <url xmlns="">
    <loc> http://www.example.com/2</loc>
    <lastmod />
  </url>
</urlset>
""")
        self.assertEqual(list(s),
            [{'priority': '1', 'loc': 'http://www.example.com/', 'lastmod': '2009-08-16', 'changefreq': 'daily'},
             {'loc': 'http://www.example.com/2', 'lastmod': ''},
            ])

    def test_sitemap_wrong_ns2(self):
        """We have seen sitemaps with wrongs ns. Presumably, Google still works
        with these, though is not 100% confirmed"""
        s = Sitemap(b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset>
  <url xmlns="">
    <loc> http://www.example.com/</loc>
    <lastmod>2009-08-16</lastmod>
    <changefreq>daily</changefreq>
    <priority>1</priority>
  </url>
  <url xmlns="">
    <loc> http://www.example.com/2</loc>
    <lastmod />
  </url>
</urlset>
""")
        assert s.type == 'urlset'
        self.assertEqual(list(s),
            [{'priority': '1', 'loc': 'http://www.example.com/', 'lastmod': '2009-08-16', 'changefreq': 'daily'},
             {'loc': 'http://www.example.com/2', 'lastmod': ''},
            ])

    def test_sitemap_urls_from_robots(self):
        robots = """User-agent: *
Disallow: /aff/
Disallow: /wl/

# Search and shopping refining
Disallow: /s*/*facet
Disallow: /s*/*tags

# Sitemap files
Sitemap: http://example.com/sitemap.xml
Sitemap: http://example.com/sitemap-product-index.xml
Sitemap: HTTP://example.com/sitemap-uppercase.xml
Sitemap: /sitemap-relative-url.xml

# Forums
Disallow: /forum/search/
Disallow: /forum/active/
"""
        self.assertEqual(list(sitemap_urls_from_robots(robots, base_url='http://example.com')),
                         ['http://example.com/sitemap.xml',
                          'http://example.com/sitemap-product-index.xml',
                          'http://example.com/sitemap-uppercase.xml',
                          'http://example.com/sitemap-relative-url.xml'])

    def test_sitemap_blanklines(self):
        """Assert we can deal with starting blank lines before <xml> tag"""
        s = Sitemap(b"""\

<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">

<!-- cache: cached = yes name = sitemap_jspCache key = sitemap -->
<sitemap>
<loc>http://www.example.com/sitemap1.xml</loc>
<lastmod>2013-07-15</lastmod>
</sitemap>

<sitemap>
<loc>http://www.example.com/sitemap2.xml</loc>
<lastmod>2013-07-15</lastmod>
</sitemap>

<sitemap>
<loc>http://www.example.com/sitemap3.xml</loc>
<lastmod>2013-07-15</lastmod>
</sitemap>

<!-- end cache -->
</sitemapindex>
""")
        self.assertEqual(list(s), [
            {'lastmod': '2013-07-15', 'loc': 'http://www.example.com/sitemap1.xml'},
            {'lastmod': '2013-07-15', 'loc': 'http://www.example.com/sitemap2.xml'},
            {'lastmod': '2013-07-15', 'loc': 'http://www.example.com/sitemap3.xml'},
        ])

    def test_comment(self):
        s = Sitemap(b"""<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:xhtml="http://www.w3.org/1999/xhtml">
        <url>
            <loc>http://www.example.com/</loc>
            <!-- this is a comment on which the parser might raise an exception if implemented incorrectly -->
        </url>
    </urlset>""")

        self.assertEqual(list(s), [
            {'loc': 'http://www.example.com/'}
        ])

    def test_alternate(self):
        s = Sitemap(b"""<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:xhtml="http://www.w3.org/1999/xhtml">
        <url>
            <loc>http://www.example.com/english/</loc>
            <xhtml:link rel="alternate" hreflang="de"
                href="http://www.example.com/deutsch/"/>
            <xhtml:link rel="alternate" hreflang="de-ch"
                href="http://www.example.com/schweiz-deutsch/"/>
            <xhtml:link rel="alternate" hreflang="en"
                href="http://www.example.com/english/"/>
            <xhtml:link rel="alternate" hreflang="en"/><!-- wrong tag without href -->
        </url>
    </urlset>""")

        self.assertEqual(list(s), [
            {'loc': 'http://www.example.com/english/',
             'alternate': ['http://www.example.com/deutsch/', 'http://www.example.com/schweiz-deutsch/', 'http://www.example.com/english/']
            }
        ])

    def test_xml_entity_expansion(self):
        s = Sitemap(b"""<?xml version="1.0" encoding="utf-8"?>
          <!DOCTYPE foo [
          <!ELEMENT foo ANY >
          <!ENTITY xxe SYSTEM "file:///etc/passwd" >
          ]>
          <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
              <loc>http://127.0.0.1:8000/&xxe;</loc>
            </url>
          </urlset>
        """)

        self.assertEqual(list(s), [{'loc': 'http://127.0.0.1:8000/'}])


if __name__ == '__main__':
    unittest.main()

def _mocked_download_func(request, info):
    response = request.meta.get('response')
    return response() if callable(response) else response


class TestPolicyHeaderPredecence002(MixinNoReferrer, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.NoReferrerWhenDowngradePolicy'}
    resp_headers = {'Referrer-Policy': POLICY_NO_REFERRER.swapcase()}

class TestGCSFilesStore(unittest.TestCase):
    @defer.inlineCallbacks
    def test_persist(self):
        assert_gcs_environ()
        uri = os.environ.get('GCS_TEST_FILE_URI')
        if not uri:
            raise unittest.SkipTest("No GCS URI available for testing")
        data = b"TestGCSFilesStore: \xe2\x98\x83"
        buf = BytesIO(data)
        meta = {'foo': 'bar'}
        path = 'full/filename'
        store = GCSFilesStore(uri)
        yield store.persist_file(path, buf, info=None, meta=meta, headers=None)
        s = yield store.stat_file(path, info=None)
        self.assertIn('last_modified', s)
        self.assertIn('checksum', s)
        self.assertEqual(s['checksum'], 'zc2oVgXkbQr2EQdSdw3OPA==')
        u = urlparse(uri)
        content, blob = get_gcs_content_and_delete(u.hostname, u.path[1:]+path)
        self.assertEqual(content, data)
        self.assertEqual(blob.metadata, {'foo': 'bar'})
        self.assertEqual(blob.cache_control, GCSFilesStore.CACHE_CONTROL)
        self.assertEqual(blob.content_type, 'application/octet-stream')


class SomeBaseClass(object):
    pass


class TestPolicyHeaderPredecence001(MixinUnsafeUrl, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.SameOriginPolicy'}
    resp_headers = {'Referrer-Policy': POLICY_UNSAFE_URL.upper()}

class MixinStrictOrigin(object):
    scenarii = [
        # TLS or non-TLS to TLS or non-TLS: referrer origin is sent but not for downgrades
        ('https://example.com/page.html',   'https://example.com/not-page.html',    b'https://example.com/'),
        ('https://example.com/page.html',   'https://scrapy.org',                   b'https://example.com/'),
        ('http://example.com/page.html',    'http://scrapy.org',                    b'http://example.com/'),

        # downgrade: send nothing
        ('https://example.com/page.html',   'http://scrapy.org',                    None),

        # upgrade: send origin
        ('http://example.com/page.html',    'https://scrapy.org',                   b'http://example.com/'),

        # test for user/password stripping
        ('https://user:password@example.com/page.html', 'https://scrapy.org',       b'https://example.com/'),
        ('https://user:password@example.com/page.html', 'http://scrapy.org',        None),
    ]


class StatsCollectorTest(unittest.TestCase):

    def setUp(self):
        self.crawler = get_crawler(Spider)
        self.spider = self.crawler._create_spider('foo')

    def test_collector(self):
        stats = StatsCollector(self.crawler)
        self.assertEqual(stats.get_stats(), {})
        self.assertEqual(stats.get_value('anything'), None)
        self.assertEqual(stats.get_value('anything', 'default'), 'default')
        stats.set_value('test', 'value')
        self.assertEqual(stats.get_stats(), {'test': 'value'})
        stats.set_value('test2', 23)
        self.assertEqual(stats.get_stats(), {'test': 'value', 'test2': 23})
        self.assertEqual(stats.get_value('test2'), 23)
        stats.inc_value('test2')
        self.assertEqual(stats.get_value('test2'), 24)
        stats.inc_value('test2', 6)
        self.assertEqual(stats.get_value('test2'), 30)
        stats.max_value('test2', 6)
        self.assertEqual(stats.get_value('test2'), 30)
        stats.max_value('test2', 40)
        self.assertEqual(stats.get_value('test2'), 40)
        stats.max_value('test3', 1)
        self.assertEqual(stats.get_value('test3'), 1)
        stats.min_value('test2', 60)
        self.assertEqual(stats.get_value('test2'), 40)
        stats.min_value('test2', 35)
        self.assertEqual(stats.get_value('test2'), 35)
        stats.min_value('test4', 7)
        self.assertEqual(stats.get_value('test4'), 7)

    def test_dummy_collector(self):
        stats = DummyStatsCollector(self.crawler)
        self.assertEqual(stats.get_stats(), {})
        self.assertEqual(stats.get_value('anything'), None)
        self.assertEqual(stats.get_value('anything', 'default'), 'default')
        stats.set_value('test', 'value')
        stats.inc_value('v1')
        stats.max_value('v2', 100)
        stats.min_value('v3', 100)
        stats.open_spider('a')
        stats.set_value('test', 'value', spider=self.spider)
        self.assertEqual(stats.get_stats(), {})
        self.assertEqual(stats.get_stats('a'), {})

class TestRequestMetaSrictOrigin(MixinStrictOrigin, TestRefererMiddleware):
    req_meta = {'referrer_policy': POLICY_STRICT_ORIGIN}


class MemoizedMethodTest(unittest.TestCase):
    def test_memoizemethod_noargs(self):
        class A(object):

            @memoizemethod_noargs
            def cached(self):
                return object()

            def noncached(self):
                return object()

        a = A()
        one = a.cached()
        two = a.cached()
        three = a.noncached()
        assert one is two
        assert one is not three

