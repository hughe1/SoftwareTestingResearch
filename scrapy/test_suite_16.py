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


class BenchCommandTest(CommandTest):

    def test_run(self):
        p = self.proc('bench', '-s', 'LOGSTATS_INTERVAL=0.001',
                '-s', 'CLOSESPIDER_TIMEOUT=0.01')
        log = to_native_str(p.stderr.read())
        self.assertIn('INFO: Crawled', log)
        self.assertNotIn('Unhandled Error', log)

class CustomPythonOrgPolicy(ReferrerPolicy):
    """
    A dummy policy that returns referrer as http(s)://python.org
    depending on the scheme of the target URL.
    """
    def referrer(self, response, request):
        scheme = urlparse(request).scheme
        if scheme == 'https':
            return b'https://python.org/'
        elif scheme == 'http':
            return b'http://python.org/'


class TestItem(NameItem):
    url = Field()
    summary = Field()


class TestRequestMetaUnsafeUrl(MixinUnsafeUrl, TestRefererMiddleware):
    req_meta = {'referrer_policy': POLICY_UNSAFE_URL}


class CrawlerLoggingTestCase(unittest.TestCase):
    def test_no_root_handler_installed(self):
        handler = get_scrapy_root_handler()
        if handler is not None:
            logging.root.removeHandler(handler)

        class MySpider(scrapy.Spider):
            name = 'spider'

        crawler = Crawler(MySpider, {})
        assert get_scrapy_root_handler() is None

    def test_spider_custom_settings_log_level(self):
        with tempfile.NamedTemporaryFile() as log_file:
            class MySpider(scrapy.Spider):
                name = 'spider'
                custom_settings = {
                    'LOG_LEVEL': 'INFO',
                    'LOG_FILE': log_file.name,
                }

            configure_logging()
            self.assertEqual(get_scrapy_root_handler().level, logging.DEBUG)
            crawler = Crawler(MySpider, {})
            self.assertEqual(get_scrapy_root_handler().level, logging.INFO)
            info_count = crawler.stats.get_value('log_count/INFO')
            logging.debug('debug message')
            logging.info('info message')
            logging.warning('warning message')
            logging.error('error message')
            logged = log_file.read().decode('utf8')
        self.assertNotIn('debug message', logged)
        self.assertIn('info message', logged)
        self.assertIn('warning message', logged)
        self.assertIn('error message', logged)
        self.assertEqual(crawler.stats.get_value('log_count/ERROR'), 1)
        self.assertEqual(crawler.stats.get_value('log_count/WARNING'), 1)
        self.assertEqual(
            crawler.stats.get_value('log_count/INFO') - info_count, 1)
        self.assertEqual(crawler.stats.get_value('log_count/DEBUG', 0), 0)


class BenchCommandTest(CommandTest):

    def test_run(self):
        p = self.proc('bench', '-s', 'LOGSTATS_INTERVAL=0.001',
                '-s', 'CLOSESPIDER_TIMEOUT=0.01')
        log = to_native_str(p.stderr.read())
        self.assertIn('INFO: Crawled', log)
        self.assertNotIn('Unhandled Error', log)

def _test_data(formats):
    uncompressed_body = get_testdata('compressed', 'feed-sample1.xml')
    test_responses = {}
    for format in formats:
        body = get_testdata('compressed', 'feed-sample1.' + format)
        test_responses[format] = Response('http://foo.com/bar', body=body)
    return uncompressed_body, test_responses


class TestSettingsUnsafeUrl(MixinUnsafeUrl, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.UnsafeUrlPolicy'}


class SpiderTest(unittest.TestCase):

    spider_class = Spider

    def setUp(self):
        warnings.simplefilter("always")

    def tearDown(self):
        warnings.resetwarnings()

    def test_base_spider(self):
        spider = self.spider_class("example.com")
        self.assertEqual(spider.name, 'example.com')
        self.assertEqual(spider.start_urls, [])

    def test_start_requests(self):
        spider = self.spider_class('example.com')
        start_requests = spider.start_requests()
        self.assertTrue(inspect.isgenerator(start_requests))
        self.assertEqual(list(start_requests), [])

    def test_spider_args(self):
        """Constructor arguments are assigned to spider attributes"""
        spider = self.spider_class('example.com', foo='bar')
        self.assertEqual(spider.foo, 'bar')

    def test_spider_without_name(self):
        """Constructor arguments are assigned to spider attributes"""
        self.assertRaises(ValueError, self.spider_class)
        self.assertRaises(ValueError, self.spider_class, somearg='foo')

    def test_deprecated_set_crawler_method(self):
        spider = self.spider_class('example.com')
        crawler = get_crawler()
        with warnings.catch_warnings(record=True) as w:
            spider.set_crawler(crawler)
            self.assertIn("set_crawler", str(w[0].message))
            self.assertTrue(hasattr(spider, 'crawler'))
            self.assertIs(spider.crawler, crawler)
            self.assertTrue(hasattr(spider, 'settings'))
            self.assertIs(spider.settings, crawler.settings)

    def test_from_crawler_crawler_and_settings_population(self):
        crawler = get_crawler()
        spider = self.spider_class.from_crawler(crawler, 'example.com')
        self.assertTrue(hasattr(spider, 'crawler'))
        self.assertIs(spider.crawler, crawler)
        self.assertTrue(hasattr(spider, 'settings'))
        self.assertIs(spider.settings, crawler.settings)

    def test_from_crawler_init_call(self):
        with mock.patch.object(self.spider_class, '__init__',
                               return_value=None) as mock_init:
            self.spider_class.from_crawler(get_crawler(), 'example.com',
                                           foo='bar')
            mock_init.assert_called_once_with('example.com', foo='bar')

    def test_closed_signal_call(self):
        class TestSpider(self.spider_class):
            closed_called = False

            def closed(self, reason):
                self.closed_called = True

        crawler = get_crawler()
        spider = TestSpider.from_crawler(crawler, 'example.com')
        crawler.signals.send_catch_log(signal=signals.spider_opened,
                                       spider=spider)
        crawler.signals.send_catch_log(signal=signals.spider_closed,
                                       spider=spider, reason=None)
        self.assertTrue(spider.closed_called)

    def test_update_settings(self):
        spider_settings = {'TEST1': 'spider', 'TEST2': 'spider'}
        project_settings = {'TEST1': 'project', 'TEST3': 'project'}
        self.spider_class.custom_settings = spider_settings
        settings = Settings(project_settings, priority='project')

        self.spider_class.update_settings(settings)
        self.assertEqual(settings.get('TEST1'), 'spider')
        self.assertEqual(settings.get('TEST2'), 'spider')
        self.assertEqual(settings.get('TEST3'), 'project')

    def test_logger(self):
        spider = self.spider_class('example.com')
        with LogCapture() as l:
            spider.logger.info('test log msg')
        l.check(('example.com', 'INFO', 'test log msg'))

        record = l.records[0]
        self.assertIn('spider', record.__dict__)
        self.assertIs(record.spider, spider)

    def test_log(self):
        spider = self.spider_class('example.com')
        with mock.patch('scrapy.spiders.Spider.logger') as mock_logger:
            spider.log('test log msg', 'INFO')
        mock_logger.log.assert_called_once_with('INFO', 'test log msg')


class FilesystemStorageTest(DefaultStorageTest):

    storage_class = 'scrapy.extensions.httpcache.FilesystemCacheStorage'

class SpiderLoaderWithWrongInterface(object):

    def unneeded_method(self):
        pass


class FilesPipelineTestCaseFields(unittest.TestCase):

    def test_item_fields_default(self):
        class TestItem(Item):
            name = Field()
            file_urls = Field()
            files = Field()

        for cls in TestItem, dict:
            url = 'http://www.example.com/files/1.txt'
            item = cls({'name': 'item1', 'file_urls': [url]})
            pipeline = FilesPipeline.from_settings(Settings({'FILES_STORE': 's3://example/files/'}))
            requests = list(pipeline.get_media_requests(item, None))
            self.assertEqual(requests[0].url, url)
            results = [(True, {'url': url})]
            pipeline.item_completed(results, item, None)
            self.assertEqual(item['files'], [results[0][1]])

    def test_item_fields_override_settings(self):
        class TestItem(Item):
            name = Field()
            files = Field()
            stored_file = Field()

        for cls in TestItem, dict:
            url = 'http://www.example.com/files/1.txt'
            item = cls({'name': 'item1', 'files': [url]})
            pipeline = FilesPipeline.from_settings(Settings({
                'FILES_STORE': 's3://example/files/',
                'FILES_URLS_FIELD': 'files',
                'FILES_RESULT_FIELD': 'stored_file'
            }))
            requests = list(pipeline.get_media_requests(item, None))
            self.assertEqual(requests[0].url, url)
            results = [(True, {'url': url})]
            pipeline.item_completed(results, item, None)
            self.assertEqual(item['stored_file'], [results[0][1]])


class BaseItemExporterTest(unittest.TestCase):

    def setUp(self):
        self.i = TestItem(name=u'John\xa3', age=u'22')
        self.output = BytesIO()
        self.ie = self._get_exporter()

    def _get_exporter(self, **kwargs):
        return BaseItemExporter(**kwargs)

    def _check_output(self):
        pass

    def _assert_expected_item(self, exported_dict):
        for k, v in exported_dict.items():
            exported_dict[k] = to_unicode(v)
        self.assertEqual(self.i, exported_dict)

    def _get_nonstring_types_item(self):
        return {
            'boolean': False,
            'number': 22,
            'time': datetime(2015, 1, 1, 1, 1, 1),
            'float': 3.14,
        }

    def assertItemExportWorks(self, item):
        self.ie.start_exporting()
        try:
            self.ie.export_item(item)
        except NotImplementedError:
            if self.ie.__class__ is not BaseItemExporter:
                raise
        self.ie.finish_exporting()
        self._check_output()

    def test_export_item(self):
        self.assertItemExportWorks(self.i)

    def test_export_dict_item(self):
        self.assertItemExportWorks(dict(self.i))

    def test_serialize_field(self):
        res = self.ie.serialize_field(self.i.fields['name'], 'name', self.i['name'])
        self.assertEqual(res, u'John\xa3')

        res = self.ie.serialize_field(self.i.fields['age'], 'age', self.i['age'])
        self.assertEqual(res, u'22')

    def test_fields_to_export(self):
        ie = self._get_exporter(fields_to_export=['name'])
        self.assertEqual(list(ie._get_serialized_fields(self.i)), [('name', u'John\xa3')])

        ie = self._get_exporter(fields_to_export=['name'], encoding='latin-1')
        _, name = list(ie._get_serialized_fields(self.i))[0]
        assert isinstance(name, six.text_type)
        self.assertEqual(name, u'John\xa3')

    def test_field_custom_serializer(self):
        def custom_serializer(value):
            return str(int(value) + 2)

        class CustomFieldItem(Item):
            name = Field()
            age = Field(serializer=custom_serializer)

        i = CustomFieldItem(name=u'John\xa3', age=u'22')

        ie = self._get_exporter()
        self.assertEqual(ie.serialize_field(i.fields['name'], 'name', i['name']), u'John\xa3')
        self.assertEqual(ie.serialize_field(i.fields['age'], 'age', i['age']), '24')


class BaseCrawlerTest(unittest.TestCase):

    def assertOptionIsDefault(self, settings, key):
        self.assertIsInstance(settings, Settings)
        self.assertEqual(settings[key], getattr(default_settings, key))


class SelectJmesTestCase(unittest.TestCase):
        test_list_equals = {
            'simple': ('foo.bar', {"foo": {"bar": "baz"}}, "baz"),
            'invalid': ('foo.bar.baz', {"foo": {"bar": "baz"}}, None),
            'top_level': ('foo', {"foo": {"bar": "baz"}}, {"bar": "baz"}),
            'double_vs_single_quote_string': ('foo.bar', {"foo": {"bar": "baz"}}, "baz"),
            'dict': (
                'foo.bar[*].name',
                {"foo": {"bar": [{"name": "one"}, {"name": "two"}]}},
                ['one', 'two']
            ),
            'list': ('[1]', [1, 2], 2)
        }

        def test_output(self):
            for l in self.test_list_equals:
                expr, test_list, expected = self.test_list_equals[l]
                test = SelectJmes(expr)(test_list)
                self.assertEqual(
                    test,
                    expected,
                    msg='test "{}" got {} expected {}'.format(l, test, expected)
                )


if __name__ == "__main__":
    unittest.main()

class StartprojectTemplatesTest(ProjectTest):

    def setUp(self):
        super(StartprojectTemplatesTest, self).setUp()
        self.tmpl = join(self.temp_path, 'templates')
        self.tmpl_proj = join(self.tmpl, 'project')

    def test_startproject_template_override(self):
        copytree(join(scrapy.__path__[0], 'templates'), self.tmpl)
        with open(join(self.tmpl_proj, 'root_template'), 'w'):
            pass
        assert exists(join(self.tmpl_proj, 'root_template'))

        args = ['--set', 'TEMPLATES_DIR=%s' % self.tmpl]
        p = self.proc('startproject', self.project_name, *args)
        out = to_native_str(retry_on_eintr(p.stdout.read))
        self.assertIn("New Scrapy project %r, using template directory" % self.project_name, out)
        self.assertIn(self.tmpl_proj, out)
        assert exists(join(self.proj_path, 'root_template'))


class TestPolicyHeaderPredecence002(MixinNoReferrer, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.NoReferrerWhenDowngradePolicy'}
    resp_headers = {'Referrer-Policy': POLICY_NO_REFERRER.swapcase()}

class HttpobjUtilsTest(unittest.TestCase):

    def test_urlparse_cached(self):
        url = "http://www.example.com/index.html"
        request1 = Request(url)
        request2 = Request(url)
        req1a = urlparse_cached(request1)
        req1b = urlparse_cached(request1)
        req2 = urlparse_cached(request2)
        urlp = urlparse(url)

        assert req1a == req2
        assert req1a == urlp
        assert req1a is req1b
        assert req1a is not req2
        assert req1a is not req2


if __name__ == "__main__":
    unittest.main()

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

class TestRequestMetaStrictOriginWhenCrossOrigin(MixinStrictOriginWhenCrossOrigin, TestRefererMiddleware):
    req_meta = {'referrer_policy': POLICY_STRICT_ORIGIN_WHEN_CROSS_ORIGIN}


class Foo(trackref.object_ref):
    pass


class UpdateClassPathTest(unittest.TestCase):

    def test_old_path_gets_fixed(self):
        with warnings.catch_warnings(record=True) as w:
            output = update_classpath('scrapy.contrib.debug.Debug')
        self.assertEqual(output, 'scrapy.extensions.debug.Debug')
        self.assertEqual(len(w), 1)
        self.assertIn("scrapy.contrib.debug.Debug", str(w[0].message))
        self.assertIn("scrapy.extensions.debug.Debug", str(w[0].message))

    def test_sorted_replacement(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ScrapyDeprecationWarning)
            output = update_classpath('scrapy.contrib.pipeline.Pipeline')
        self.assertEqual(output, 'scrapy.pipelines.Pipeline')

    def test_unmatched_path_stays_the_same(self):
        with warnings.catch_warnings(record=True) as w:
            output = update_classpath('scrapy.unmatched.Path')
        self.assertEqual(output, 'scrapy.unmatched.Path')
        self.assertEqual(len(w), 0)

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

class CrawlTestCase(TestCase):

    def setUp(self):
        self.mockserver = MockServer()
        self.mockserver.__enter__()
        self.runner = CrawlerRunner()

    def tearDown(self):
        self.mockserver.__exit__(None, None, None)

    @defer.inlineCallbacks
    def test_follow_all(self):
        crawler = self.runner.create_crawler(FollowAllSpider)
        yield crawler.crawl()
        self.assertEqual(len(crawler.spider.urls_visited), 11)  # 10 + start_url

    @defer.inlineCallbacks
    def test_delay(self):
        # short to long delays
        yield self._test_delay(0.2, False)
        yield self._test_delay(1, False)
        # randoms
        yield self._test_delay(0.2, True)
        yield self._test_delay(1, True)

    @defer.inlineCallbacks
    def _test_delay(self, delay, randomize):
        settings = {"DOWNLOAD_DELAY": delay, 'RANDOMIZE_DOWNLOAD_DELAY': randomize}
        crawler = CrawlerRunner(settings).create_crawler(FollowAllSpider)
        yield crawler.crawl(maxlatency=delay * 2)
        t = crawler.spider.times
        totaltime = t[-1] - t[0]
        avgd = totaltime / (len(t) - 1)
        tolerance = 0.6 if randomize else 0.2
        self.assertTrue(avgd > delay * (1 - tolerance),
                        "download delay too small: %s" % avgd)

    @defer.inlineCallbacks
    def test_timeout_success(self):
        crawler = self.runner.create_crawler(DelaySpider)
        yield crawler.crawl(n=0.5)
        self.assertTrue(crawler.spider.t1 > 0)
        self.assertTrue(crawler.spider.t2 > 0)
        self.assertTrue(crawler.spider.t2 > crawler.spider.t1)

    @defer.inlineCallbacks
    def test_timeout_failure(self):
        crawler = CrawlerRunner({"DOWNLOAD_TIMEOUT": 0.35}).create_crawler(DelaySpider)
        yield crawler.crawl(n=0.5)
        self.assertTrue(crawler.spider.t1 > 0)
        self.assertTrue(crawler.spider.t2 == 0)
        self.assertTrue(crawler.spider.t2_err > 0)
        self.assertTrue(crawler.spider.t2_err > crawler.spider.t1)
        # server hangs after receiving response headers
        yield crawler.crawl(n=0.5, b=1)
        self.assertTrue(crawler.spider.t1 > 0)
        self.assertTrue(crawler.spider.t2 == 0)
        self.assertTrue(crawler.spider.t2_err > 0)
        self.assertTrue(crawler.spider.t2_err > crawler.spider.t1)

    @defer.inlineCallbacks
    def test_retry_503(self):
        crawler = self.runner.create_crawler(SimpleSpider)
        with LogCapture() as l:
            yield crawler.crawl("http://localhost:8998/status?n=503")
        self._assert_retried(l)

    @defer.inlineCallbacks
    def test_retry_conn_failed(self):
        crawler = self.runner.create_crawler(SimpleSpider)
        with LogCapture() as l:
            yield crawler.crawl("http://localhost:65432/status?n=503")
        self._assert_retried(l)

    @defer.inlineCallbacks
    def test_retry_dns_error(self):
        crawler = self.runner.create_crawler(SimpleSpider)
        with LogCapture() as l:
            # try to fetch the homepage of a non-existent domain
            yield crawler.crawl("http://dns.resolution.invalid./")
        self._assert_retried(l)

    @defer.inlineCallbacks
    def test_start_requests_bug_before_yield(self):
        with LogCapture('scrapy', level=logging.ERROR) as l:
            crawler = self.runner.create_crawler(BrokenStartRequestsSpider)
            yield crawler.crawl(fail_before_yield=1)

        self.assertEqual(len(l.records), 1)
        record = l.records[0]
        self.assertIsNotNone(record.exc_info)
        self.assertIs(record.exc_info[0], ZeroDivisionError)

    @defer.inlineCallbacks
    def test_start_requests_bug_yielding(self):
        with LogCapture('scrapy', level=logging.ERROR) as l:
            crawler = self.runner.create_crawler(BrokenStartRequestsSpider)
            yield crawler.crawl(fail_yielding=1)

        self.assertEqual(len(l.records), 1)
        record = l.records[0]
        self.assertIsNotNone(record.exc_info)
        self.assertIs(record.exc_info[0], ZeroDivisionError)

    @defer.inlineCallbacks
    def test_start_requests_lazyness(self):
        settings = {"CONCURRENT_REQUESTS": 1}
        crawler = CrawlerRunner(settings).create_crawler(BrokenStartRequestsSpider)
        yield crawler.crawl()
        #self.assertTrue(False, crawler.spider.seedsseen)
        #self.assertTrue(crawler.spider.seedsseen.index(None) < crawler.spider.seedsseen.index(99),
        #                crawler.spider.seedsseen)

    @defer.inlineCallbacks
    def test_start_requests_dupes(self):
        settings = {"CONCURRENT_REQUESTS": 1}
        crawler = CrawlerRunner(settings).create_crawler(DuplicateStartRequestsSpider)
        yield crawler.crawl(dont_filter=True, distinct_urls=2, dupe_factor=3)
        self.assertEqual(crawler.spider.visited, 6)

        yield crawler.crawl(dont_filter=False, distinct_urls=3, dupe_factor=4)
        self.assertEqual(crawler.spider.visited, 3)

    @defer.inlineCallbacks
    def test_unbounded_response(self):
        # Completeness of responses without Content-Length or Transfer-Encoding
        # can not be determined, we treat them as valid but flagged as "partial"
        from six.moves.urllib.parse import urlencode
        query = urlencode({'raw': '''\
HTTP/1.1 200 OK
Server: Apache-Coyote/1.1
X-Powered-By: Servlet 2.4; JBoss-4.2.3.GA (build: SVNTag=JBoss_4_2_3_GA date=200807181417)/JBossWeb-2.0
Set-Cookie: JSESSIONID=08515F572832D0E659FD2B0D8031D75F; Path=/
Pragma: no-cache
Expires: Thu, 01 Jan 1970 00:00:00 GMT
Cache-Control: no-cache
Cache-Control: no-store
Content-Type: text/html;charset=UTF-8
Content-Language: en
Date: Tue, 27 Aug 2013 13:05:05 GMT
Connection: close

foo body
with multiples lines
'''})
        crawler = self.runner.create_crawler(SimpleSpider)
        with LogCapture() as l:
            yield crawler.crawl("http://localhost:8998/raw?{0}".format(query))
        self.assertEqual(str(l).count("Got response 200"), 1)

    @defer.inlineCallbacks
    def test_retry_conn_lost(self):
        # connection lost after receiving data
        crawler = self.runner.create_crawler(SimpleSpider)
        with LogCapture() as l:
            yield crawler.crawl("http://localhost:8998/drop?abort=0")
        self._assert_retried(l)

    @defer.inlineCallbacks
    def test_retry_conn_aborted(self):
        # connection lost before receiving data
        crawler = self.runner.create_crawler(SimpleSpider)
        with LogCapture() as l:
            yield crawler.crawl("http://localhost:8998/drop?abort=1")
        self._assert_retried(l)

    def _assert_retried(self, log):
        self.assertEqual(str(log).count("Retrying"), 2)
        self.assertEqual(str(log).count("Gave up retrying"), 1)

    @defer.inlineCallbacks
    def test_referer_header(self):
        """Referer header is set by RefererMiddleware unless it is already set"""
        req0 = Request('http://localhost:8998/echo?headers=1&body=0', dont_filter=1)
        req1 = req0.replace()
        req2 = req0.replace(headers={'Referer': None})
        req3 = req0.replace(headers={'Referer': 'http://example.com'})
        req0.meta['next'] = req1
        req1.meta['next'] = req2
        req2.meta['next'] = req3
        crawler = self.runner.create_crawler(SingleRequestSpider)
        yield crawler.crawl(seed=req0)
        # basic asserts in case of weird communication errors
        self.assertIn('responses', crawler.spider.meta)
        self.assertNotIn('failures', crawler.spider.meta)
        # start requests doesn't set Referer header
        echo0 = json.loads(to_unicode(crawler.spider.meta['responses'][2].body))
        self.assertNotIn('Referer', echo0['headers'])
        # following request sets Referer to start request url
        echo1 = json.loads(to_unicode(crawler.spider.meta['responses'][1].body))
        self.assertEqual(echo1['headers'].get('Referer'), [req0.url])
        # next request avoids Referer header
        echo2 = json.loads(to_unicode(crawler.spider.meta['responses'][2].body))
        self.assertNotIn('Referer', echo2['headers'])
        # last request explicitly sets a Referer header
        echo3 = json.loads(to_unicode(crawler.spider.meta['responses'][3].body))
        self.assertEqual(echo3['headers'].get('Referer'), ['http://example.com'])

    @defer.inlineCallbacks
    def test_engine_status(self):
        from scrapy.utils.engine import get_engine_status
        est = []

        def cb(response):
            est.append(get_engine_status(crawler.engine))

        crawler = self.runner.create_crawler(SingleRequestSpider)
        yield crawler.crawl(seed='http://localhost:8998/', callback_func=cb)
        self.assertEqual(len(est), 1, est)
        s = dict(est[0])
        self.assertEqual(s['engine.spider.name'], crawler.spider.name)
        self.assertEqual(s['len(engine.scraper.slot.active)'], 1)

    @defer.inlineCallbacks
    def test_graceful_crawl_error_handling(self):
        """
        Test whether errors happening anywhere in Crawler.crawl() are properly
        reported (and not somehow swallowed) after a graceful engine shutdown.
        The errors should not come from within Scrapy's core but from within
        spiders/middlewares/etc., e.g. raised in Spider.start_requests(),
        SpiderMiddleware.process_start_requests(), etc.
        """

        class TestError(Exception):
            pass

        class FaultySpider(SimpleSpider):
            def start_requests(self):
                raise TestError

        crawler = self.runner.create_crawler(FaultySpider)
        yield self.assertFailure(crawler.crawl(), TestError)
        self.assertFalse(crawler.crawling)

    @defer.inlineCallbacks
    def test_open_spider_error_on_faulty_pipeline(self):
        settings = {
            "ITEM_PIPELINES": {
                "tests.pipelines.ZeroDivisionErrorPipeline": 300,
            }
        }
        crawler = CrawlerRunner(settings).create_crawler(SimpleSpider)
        yield self.assertFailure(
            self.runner.crawl(crawler, "http://localhost:8998/status?n=200"),
            ZeroDivisionError)
        self.assertFalse(crawler.crawling)

    @defer.inlineCallbacks
    def test_crawlerrunner_accepts_crawler(self):
        crawler = self.runner.create_crawler(SimpleSpider)
        with LogCapture() as log:
            yield self.runner.crawl(crawler, "http://localhost:8998/status?n=200")
        self.assertIn("Got response 200", str(log))

    @defer.inlineCallbacks
    def test_crawl_multiple(self):
        self.runner.crawl(SimpleSpider, "http://localhost:8998/status?n=200")
        self.runner.crawl(SimpleSpider, "http://localhost:8998/status?n=503")

        with LogCapture() as log:
            yield self.runner.join()

        self._assert_retried(log)
        self.assertIn("Got response 200", str(log))

class Foo(trackref.object_ref):
    pass


class TrackrefTestCase(unittest.TestCase):

    def setUp(self):
        trackref.live_refs.clear()

    def test_format_live_refs(self):
        o1 = Foo()  # NOQA
        o2 = Bar()  # NOQA
        o3 = Foo()  # NOQA
        self.assertEqual(
            trackref.format_live_refs(),
            '''\
Live References

Bar                                 1   oldest: 0s ago
Foo                                 2   oldest: 0s ago
''')

        self.assertEqual(
            trackref.format_live_refs(ignore=Foo),
            '''\
Live References

Bar                                 1   oldest: 0s ago
''')

    @mock.patch('sys.stdout', new_callable=six.StringIO)
    def test_print_live_refs_empty(self, stdout):
        trackref.print_live_refs()
        self.assertEqual(stdout.getvalue(), 'Live References\n\n\n')

    @mock.patch('sys.stdout', new_callable=six.StringIO)
    def test_print_live_refs_with_objects(self, stdout):
        o1 = Foo()  # NOQA
        trackref.print_live_refs()
        self.assertEqual(stdout.getvalue(), '''\
Live References

Foo                                 1   oldest: 0s ago\n\n''')

    def test_get_oldest(self):
        o1 = Foo()  # NOQA
        o2 = Bar()  # NOQA
        o3 = Foo()  # NOQA
        self.assertIs(trackref.get_oldest('Foo'), o1)
        self.assertIs(trackref.get_oldest('Bar'), o2)
        self.assertIsNone(trackref.get_oldest('XXX'))

    def test_iter_all(self):
        o1 = Foo()  # NOQA
        o2 = Bar()  # NOQA
        o3 = Foo()  # NOQA
        self.assertEqual(
            set(trackref.iter_all('Foo')),
            {o1, o3},
        )

class TestLoader(ItemLoader):
    default_item_class = TestItem
    name_out = staticmethod(_test_procesor)

def nonserializable_object_test(self):
    try:
        pickle.dumps(lambda x: x)
    except Exception:
        # Trigger Twisted bug #7989
        import twisted.persisted.styles  # NOQA
        q = self.queue()
        self.assertRaises(ValueError, q.push, lambda x: x)
    else:
        # Use a different unpickleable object
        class A(object): pass
        a = A()
        a.__reduce__ = a.__reduce_ex__ = None
        q = self.queue()
        self.assertRaises(ValueError, q.push, a)

class MailSenderTest(unittest.TestCase):

    def test_send(self):
        mailsender = MailSender(debug=True)
        mailsender.send(to=['test@scrapy.org'], subject='subject', body='body',
                        _callback=self._catch_mail_sent)

        assert self.catched_msg

        self.assertEqual(self.catched_msg['to'], ['test@scrapy.org'])
        self.assertEqual(self.catched_msg['subject'], 'subject')
        self.assertEqual(self.catched_msg['body'], 'body')

        msg = self.catched_msg['msg']
        self.assertEqual(msg['to'], 'test@scrapy.org')
        self.assertEqual(msg['subject'], 'subject')
        self.assertEqual(msg.get_payload(), 'body')
        self.assertEqual(msg.get('Content-Type'), 'text/plain')

    def test_send_single_values_to_and_cc(self):
        mailsender = MailSender(debug=True)
        mailsender.send(to='test@scrapy.org', subject='subject', body='body',
                        cc='test@scrapy.org', _callback=self._catch_mail_sent)

    def test_send_html(self):
        mailsender = MailSender(debug=True)
        mailsender.send(to=['test@scrapy.org'], subject='subject',
                        body='<p>body</p>', mimetype='text/html',
                        _callback=self._catch_mail_sent)

        msg = self.catched_msg['msg']
        self.assertEqual(msg.get_payload(), '<p>body</p>')
        self.assertEqual(msg.get('Content-Type'), 'text/html')

    def test_send_attach(self):
        attach = BytesIO()
        attach.write(b'content')
        attach.seek(0)
        attachs = [('attachment', 'text/plain', attach)]

        mailsender = MailSender(debug=True)
        mailsender.send(to=['test@scrapy.org'], subject='subject', body='body',
                       attachs=attachs, _callback=self._catch_mail_sent)

        assert self.catched_msg
        self.assertEqual(self.catched_msg['to'], ['test@scrapy.org'])
        self.assertEqual(self.catched_msg['subject'], 'subject')
        self.assertEqual(self.catched_msg['body'], 'body')

        msg = self.catched_msg['msg']
        self.assertEqual(msg['to'], 'test@scrapy.org')
        self.assertEqual(msg['subject'], 'subject')

        payload = msg.get_payload()
        assert isinstance(payload, list)
        self.assertEqual(len(payload), 2)

        text, attach = payload
        self.assertEqual(text.get_payload(decode=True), b'body')
        self.assertEqual(text.get_charset(), Charset('us-ascii'))
        self.assertEqual(attach.get_payload(decode=True), b'content')

    def _catch_mail_sent(self, **kwargs):
        self.catched_msg = dict(**kwargs)

    def test_send_utf8(self):
        subject = u'sübjèçt'
        body = u'bödÿ-àéïöñß'
        mailsender = MailSender(debug=True)
        mailsender.send(to=['test@scrapy.org'], subject=subject, body=body,
                        charset='utf-8', _callback=self._catch_mail_sent)

        assert self.catched_msg
        self.assertEqual(self.catched_msg['subject'], subject)
        self.assertEqual(self.catched_msg['body'], body)

        msg = self.catched_msg['msg']
        self.assertEqual(msg['subject'], subject)
        self.assertEqual(msg.get_payload(), body)
        self.assertEqual(msg.get_charset(), Charset('utf-8'))
        self.assertEqual(msg.get('Content-Type'), 'text/plain; charset="utf-8"')

    def test_send_attach_utf8(self):
        subject = u'sübjèçt'
        body = u'bödÿ-àéïöñß'
        attach = BytesIO()
        attach.write(body.encode('utf-8'))
        attach.seek(0)
        attachs = [('attachment', 'text/plain', attach)]

        mailsender = MailSender(debug=True)
        mailsender.send(to=['test@scrapy.org'], subject=subject, body=body,
                        attachs=attachs, charset='utf-8',
                        _callback=self._catch_mail_sent)

        assert self.catched_msg
        self.assertEqual(self.catched_msg['subject'], subject)
        self.assertEqual(self.catched_msg['body'], body)

        msg = self.catched_msg['msg']
        self.assertEqual(msg['subject'], subject)
        self.assertEqual(msg.get_charset(), Charset('utf-8'))
        self.assertEqual(msg.get('Content-Type'),
                         'multipart/mixed; charset="utf-8"')

        payload = msg.get_payload()
        assert isinstance(payload, list)
        self.assertEqual(len(payload), 2)

        text, attach = payload
        self.assertEqual(text.get_payload(decode=True).decode('utf-8'), body)
        self.assertEqual(text.get_charset(), Charset('utf-8'))
        self.assertEqual(attach.get_payload(decode=True).decode('utf-8'), body)

if __name__ == "__main__":
    unittest.main()

class LxmlLinkExtractorTestCase(Base.LinkExtractorTestCase):
    extractor_cls = LxmlLinkExtractor

    def test_link_wrong_href(self):
        html = b"""
        <a href="http://example.org/item1.html">Item 1</a>
        <a href="http://[example.org/item2.html">Item 2</a>
        <a href="http://example.org/item3.html">Item 3</a>
        """
        response = HtmlResponse("http://example.org/index.html", body=html)
        lx = self.extractor_cls()
        self.assertEqual([link for link in lx.extract_links(response)], [
            Link(url='http://example.org/item1.html', text=u'Item 1', nofollow=False),
            Link(url='http://example.org/item3.html', text=u'Item 3', nofollow=False),
        ])

    @pytest.mark.xfail
    def test_restrict_xpaths_with_html_entities(self):
        super(LxmlLinkExtractorTestCase, self).test_restrict_xpaths_with_html_entities()


class MixinOrigin(object):
    scenarii = [
        # TLS or non-TLS to TLS or non-TLS: referrer origin is sent (yes, even for downgrades)
        ('https://example.com/page.html',   'https://example.com/not-page.html',    b'https://example.com/'),
        ('https://example.com/page.html',   'https://scrapy.org',                   b'https://example.com/'),
        ('https://example.com/page.html',   'http://scrapy.org',                    b'https://example.com/'),
        ('http://example.com/page.html',    'http://scrapy.org',                    b'http://example.com/'),

        # test for user/password stripping
        ('https://user:password@example.com/page.html', 'http://scrapy.org', b'https://example.com/'),
    ]

