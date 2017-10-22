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


class TestSpider(Spider):
    http_user = 'foo'
    http_pass = 'bar'


class TestRequestMetaSameOrigin(MixinSameOrigin, TestRefererMiddleware):
    req_meta = {'referrer_policy': POLICY_SAME_ORIGIN}


class RequestSerializationTest(unittest.TestCase):

    def setUp(self):
        self.spider = TestSpider()

    def test_basic(self):
        r = Request("http://www.example.com")
        self._assert_serializes_ok(r)

    def test_all_attributes(self):
        r = Request("http://www.example.com",
            callback=self.spider.parse_item,
            errback=self.spider.handle_error,
            method="POST",
            body=b"some body",
            headers={'content-encoding': 'text/html; charset=latin-1'},
            cookies={'currency': u'руб'},
            encoding='latin-1',
            priority=20,
            meta={'a': 'b'},
            flags=['testFlag'])
        self._assert_serializes_ok(r, spider=self.spider)

    def test_latin1_body(self):
        r = Request("http://www.example.com", body=b"\xa3")
        self._assert_serializes_ok(r)

    def test_utf8_body(self):
        r = Request("http://www.example.com", body=b"\xc2\xa3")
        self._assert_serializes_ok(r)

    def _assert_serializes_ok(self, request, spider=None):
        d = request_to_dict(request, spider=spider)
        request2 = request_from_dict(d, spider=spider)
        self._assert_same_request(request, request2)

    def _assert_same_request(self, r1, r2):
        self.assertEqual(r1.__class__, r2.__class__)
        self.assertEqual(r1.url, r2.url)
        self.assertEqual(r1.callback, r2.callback)
        self.assertEqual(r1.errback, r2.errback)
        self.assertEqual(r1.method, r2.method)
        self.assertEqual(r1.body, r2.body)
        self.assertEqual(r1.headers, r2.headers)
        self.assertEqual(r1.cookies, r2.cookies)
        self.assertEqual(r1.meta, r2.meta)
        self.assertEqual(r1._encoding, r2._encoding)
        self.assertEqual(r1.priority, r2.priority)
        self.assertEqual(r1.dont_filter, r2.dont_filter)
        self.assertEqual(r1.flags, r2.flags)

    def test_request_class(self):
        r = FormRequest("http://www.example.com")
        self._assert_serializes_ok(r, spider=self.spider)
        r = CustomRequest("http://www.example.com")
        self._assert_serializes_ok(r, spider=self.spider)

    def test_callback_serialization(self):
        r = Request("http://www.example.com", callback=self.spider.parse_item,
                    errback=self.spider.handle_error)
        self._assert_serializes_ok(r, spider=self.spider)

    def test_unserializable_callback1(self):
        r = Request("http://www.example.com", callback=lambda x: x)
        self.assertRaises(ValueError, request_to_dict, r)
        self.assertRaises(ValueError, request_to_dict, r, spider=self.spider)

    def test_unserializable_callback2(self):
        r = Request("http://www.example.com", callback=self.spider.parse_item)
        self.assertRaises(ValueError, request_to_dict, r)


class DeprecatedPydispatchTest(unittest.TestCase):
    def test_import_xlib_pydispatch_show_warning(self):
        with warnings.catch_warnings(record=True) as w:
            from scrapy.xlib import pydispatch
            reload_module(pydispatch)
        self.assertIn('Importing from scrapy.xlib.pydispatch is deprecated',
                      str(w[0].message))

class BuildComponentListTest(unittest.TestCase):

    def test_build_dict(self):
        d = {'one': 1, 'two': None, 'three': 8, 'four': 4}
        self.assertEqual(build_component_list(d, convert=lambda x: x),
                         ['one', 'four', 'three'])

    def test_backwards_compatible_build_dict(self):
        base = {'one': 1, 'two': 2, 'three': 3, 'five': 5, 'six': None}
        custom = {'two': None, 'three': 8, 'four': 4}
        self.assertEqual(build_component_list(base, custom,
                                              convert=lambda x: x),
                         ['one', 'four', 'five', 'three'])

    def test_return_list(self):
        custom = ['a', 'b', 'c']
        self.assertEqual(build_component_list(None, custom,
                                              convert=lambda x: x),
                         custom)

    def test_map_dict(self):
        custom = {'one': 1, 'two': 2, 'three': 3}
        self.assertEqual(build_component_list({}, custom,
                                              convert=lambda x: x.upper()),
                         ['ONE', 'TWO', 'THREE'])

    def test_map_list(self):
        custom = ['a', 'b', 'c']
        self.assertEqual(build_component_list(None, custom,
                                              lambda x: x.upper()),
                         ['A', 'B', 'C'])

    def test_duplicate_components_in_dict(self):
        duplicate_dict = {'one': 1, 'two': 2, 'ONE': 4}
        self.assertRaises(ValueError, build_component_list, {}, duplicate_dict,
                          convert=lambda x: x.lower())

    def test_duplicate_components_in_list(self):
        duplicate_list = ['a', 'b', 'a']
        self.assertRaises(ValueError, build_component_list, None,
                          duplicate_list, convert=lambda x: x)

    def test_duplicate_components_in_basesettings(self):
        # Higher priority takes precedence
        duplicate_bs = BaseSettings({'one': 1, 'two': 2}, priority=0)
        duplicate_bs.set('ONE', 4, priority=10)
        self.assertEqual(build_component_list(duplicate_bs,
                                              convert=lambda x: x.lower()),
                         ['two', 'one'])
        duplicate_bs.set('one', duplicate_bs['one'], priority=20)
        self.assertEqual(build_component_list(duplicate_bs,
                                              convert=lambda x: x.lower()),
                         ['one', 'two'])
        # Same priority raises ValueError
        duplicate_bs.set('ONE', duplicate_bs['ONE'], priority=20)
        self.assertRaises(ValueError, build_component_list, duplicate_bs,
                          convert=lambda x: x.lower())

    def test_valid_numbers(self):
        # work well with None and numeric values
        d = {'a': 10, 'b': None, 'c': 15, 'd': 5.0}
        self.assertEqual(build_component_list(d, convert=lambda x: x),
                         ['d', 'a', 'c'])
        d = {'a': 33333333333333333333, 'b': 11111111111111111111, 'c': 22222222222222222222}
        self.assertEqual(build_component_list(d, convert=lambda x: x),
                         ['b', 'c', 'a'])
        # raise exception for invalid values
        d = {'one': '5'}
        self.assertRaises(ValueError, build_component_list, {}, d, convert=lambda x: x)
        d = {'one': '1.0'}
        self.assertRaises(ValueError, build_component_list, {}, d, convert=lambda x: x)
        d = {'one': [1, 2, 3]}
        self.assertRaises(ValueError, build_component_list, {}, d, convert=lambda x: x)
        d = {'one': {'a': 'a', 'b': 2}}
        self.assertRaises(ValueError, build_component_list, {}, d, convert=lambda x: x)
        d = {'one': 'lorem ipsum',}
        self.assertRaises(ValueError, build_component_list, {}, d, convert=lambda x: x)



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


class EmptyContentTypeHeaderResource(resource.Resource):
    """
    A testing resource which renders itself as the value of request body
    without content-type header in response.
    """
    def render(self, request):
        request.setHeader("content-type", "")
        return request.content.read()


class WebClientTestCase(unittest.TestCase):
    def _listen(self, site):
        return reactor.listenTCP(0, site, interface="127.0.0.1")

    def setUp(self):
        self.tmpname = self.mktemp()
        os.mkdir(self.tmpname)
        FilePath(self.tmpname).child("file").setContent(b"0123456789")
        r = static.File(self.tmpname)
        r.putChild(b"redirect", util.Redirect(b"/file"))
        r.putChild(b"wait", ForeverTakingResource())
        r.putChild(b"error", ErrorResource())
        r.putChild(b"nolength", NoLengthResource())
        r.putChild(b"host", HostHeaderResource())
        r.putChild(b"payload", PayloadResource())
        r.putChild(b"broken", BrokenDownloadResource())
        r.putChild(b"encoding", EncodingResource())
        self.site = server.Site(r, timeout=None)
        self.wrapper = WrappingFactory(self.site)
        self.port = self._listen(self.wrapper)
        self.portno = self.port.getHost().port

    @inlineCallbacks
    def tearDown(self):
        yield self.port.stopListening()
        shutil.rmtree(self.tmpname)

    def getURL(self, path):
        return "http://127.0.0.1:%d/%s" % (self.portno, path)

    def testPayload(self):
        s = "0123456789" * 10
        return getPage(self.getURL("payload"), body=s).addCallback(
            self.assertEqual, to_bytes(s))

    def testHostHeader(self):
        # if we pass Host header explicitly, it should be used, otherwise
        # it should extract from url
        return defer.gatherResults([
            getPage(self.getURL("host")).addCallback(
                self.assertEqual, to_bytes("127.0.0.1:%d" % self.portno)),
            getPage(self.getURL("host"), headers={"Host": "www.example.com"}).addCallback(
                self.assertEqual, to_bytes("www.example.com"))])

    def test_getPage(self):
        """
        L{client.getPage} returns a L{Deferred} which is called back with
        the body of the response if the default method B{GET} is used.
        """
        d = getPage(self.getURL("file"))
        d.addCallback(self.assertEqual, b"0123456789")
        return d

    def test_getPageHead(self):
        """
        L{client.getPage} returns a L{Deferred} which is called back with
        the empty string if the method is C{HEAD} and there is a successful
        response code.
        """
        def _getPage(method):
            return getPage(self.getURL("file"), method=method)
        return defer.gatherResults([
            _getPage("head").addCallback(self.assertEqual, b""),
            _getPage("HEAD").addCallback(self.assertEqual, b"")])

    def test_timeoutNotTriggering(self):
        """
        When a non-zero timeout is passed to L{getPage} and the page is
        retrieved before the timeout period elapses, the L{Deferred} is
        called back with the contents of the page.
        """
        d = getPage(self.getURL("host"), timeout=100)
        d.addCallback(
            self.assertEqual, to_bytes("127.0.0.1:%d" % self.portno))
        return d

    def test_timeoutTriggering(self):
        """
        When a non-zero timeout is passed to L{getPage} and that many
        seconds elapse before the server responds to the request. the
        L{Deferred} is errbacked with a L{error.TimeoutError}.
        """
        finished = self.assertFailure(
            getPage(self.getURL("wait"), timeout=0.000001),
            defer.TimeoutError)
        def cleanup(passthrough):
            # Clean up the server which is hanging around not doing
            # anything.
            connected = list(six.iterkeys(self.wrapper.protocols))
            # There might be nothing here if the server managed to already see
            # that the connection was lost.
            if connected:
                connected[0].transport.loseConnection()
            return passthrough
        finished.addBoth(cleanup)
        return finished

    def testNotFound(self):
        return getPage(self.getURL('notsuchfile')).addCallback(self._cbNoSuchFile)

    def _cbNoSuchFile(self, pageData):
        self.assertIn(b'404 - No Such Resource', pageData)

    def testFactoryInfo(self):
        url = self.getURL('file')
        _, _, host, port, _ = client._parse(url)
        factory = client.ScrapyHTTPClientFactory(Request(url))
        reactor.connectTCP(to_unicode(host), port, factory)
        return factory.deferred.addCallback(self._cbFactoryInfo, factory)

    def _cbFactoryInfo(self, ignoredResult, factory):
        self.assertEqual(factory.status, b'200')
        self.assertTrue(factory.version.startswith(b'HTTP/'))
        self.assertEqual(factory.message, b'OK')
        self.assertEqual(factory.response_headers[b'content-length'], b'10')

    def testRedirect(self):
        return getPage(self.getURL("redirect")).addCallback(self._cbRedirect)

    def _cbRedirect(self, pageData):
        self.assertEqual(pageData,
                b'\n<html>\n    <head>\n        <meta http-equiv="refresh" content="0;URL=/file">\n'
                b'    </head>\n    <body bgcolor="#FFFFFF" text="#000000">\n    '
                b'<a href="/file">click here</a>\n    </body>\n</html>\n')

    def test_encoding(self):
        """ Test that non-standart body encoding matches
        Content-Encoding header """
        body = b'\xd0\x81\xd1\x8e\xd0\xaf'
        return getPage(
            self.getURL('encoding'), body=body, response_transform=lambda r: r)\
            .addCallback(self._check_Encoding, body)

    def _check_Encoding(self, response, original_body):
        content_encoding = to_unicode(response.headers[b'Content-Encoding'])
        self.assertEqual(content_encoding, EncodingResource.out_encoding)
        self.assertEqual(
            response.body.decode(content_encoding), to_unicode(original_body))

class ParseCommandTest(ProcessTest, SiteTest, CommandTest):
    command = 'parse'

    def setUp(self):
        super(ParseCommandTest, self).setUp()
        self.spider_name = 'parse_spider'
        fname = abspath(join(self.proj_mod_path, 'spiders', 'myspider.py'))
        with open(fname, 'w') as f:
            f.write("""
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule


class TestPolicyHeaderPredecence003(MixinNoReferrerWhenDowngrade, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.OriginWhenCrossOriginPolicy'}
    resp_headers = {'Referrer-Policy': POLICY_NO_REFERRER_WHEN_DOWNGRADE.title()}

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

class RunSpiderCommandTest(CommandTest):

    debug_log_spider = """
import scrapy

class SequenceExcludeTest(unittest.TestCase):

    def test_list(self):
        seq = [1, 2, 3]
        d = SequenceExclude(seq)
        self.assertIn(0, d)
        self.assertIn(4, d)
        self.assertNotIn(2, d)

    def test_range(self):
        seq = range(10, 20)
        d = SequenceExclude(seq)
        self.assertIn(5, d)
        self.assertIn(20, d)
        self.assertNotIn(15, d)

    def test_six_range(self):
        import six.moves
        seq = six.moves.range(10**3, 10**6)
        d = SequenceExclude(seq)
        self.assertIn(10**2, d)
        self.assertIn(10**7, d)
        self.assertNotIn(10**4, d)

    def test_range_step(self):
        seq = range(10, 20, 3)
        d = SequenceExclude(seq)
        are_not_in = [v for v in range(10, 20, 3) if v in d]
        self.assertEqual([], are_not_in)

        are_not_in = [v for v in range(10, 20) if v in d]
        self.assertEqual([11, 12, 14, 15, 17, 18], are_not_in)

    def test_string_seq(self):
        seq = "cde"
        d = SequenceExclude(seq)
        chars = "".join(v for v in "abcdefg" if v in d)
        self.assertEqual("abfg", chars)

    def test_stringset_seq(self):
        seq = set("cde")
        d = SequenceExclude(seq)
        chars = "".join(v for v in "abcdefg" if v in d)
        self.assertEqual("abfg", chars)

    def test_set(self):
        """Anything that is not in the supplied sequence will evaluate as 'in' the container."""
        seq = set([-3, "test", 1.1])
        d = SequenceExclude(seq)
        self.assertIn(0, d)
        self.assertIn("foo", d)
        self.assertIn(3.14, d)
        self.assertIn(set("bar"), d)

        # supplied sequence is a set, so checking for list (non)inclusion fails
        self.assertRaises(TypeError, (0, 1, 2) in d)
        self.assertRaises(TypeError, d.__contains__, ['a', 'b', 'c'])

        for v in [-3, "test", 1.1]:
            self.assertNotIn(v, d)

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


class TestOffsiteMiddleware4(TestOffsiteMiddleware3):

    def _get_spider(self):
      bad_hostname = urlparse('http:////scrapytest.org').hostname
      return dict(name='foo', allowed_domains=['scrapytest.org', None, bad_hostname])

    def test_process_spider_output(self):
      res = Response('http://scrapytest.org')
      reqs = [Request('http://scrapytest.org/1')]
      out = list(self.mw.process_spider_output(res, reqs, self.spider))
      self.assertEqual(out, reqs)

class HttpTestCase(unittest.TestCase):

    scheme = 'http'
    download_handler_cls = HTTPDownloadHandler

    # only used for HTTPS tests
    keyfile = 'keys/localhost.key'
    certfile = 'keys/localhost.crt'

    def setUp(self):
        self.tmpname = self.mktemp()
        os.mkdir(self.tmpname)
        FilePath(self.tmpname).child("file").setContent(b"0123456789")
        r = static.File(self.tmpname)
        r.putChild(b"redirect", util.Redirect(b"/file"))
        r.putChild(b"wait", ForeverTakingResource())
        r.putChild(b"hang-after-headers", ForeverTakingResource(write=True))
        r.putChild(b"nolength", NoLengthResource())
        r.putChild(b"host", HostHeaderResource())
        r.putChild(b"payload", PayloadResource())
        r.putChild(b"broken", BrokenDownloadResource())
        r.putChild(b"chunked", ChunkedResource())
        r.putChild(b"broken-chunked", BrokenChunkedResource())
        r.putChild(b"contentlength", ContentLengthHeaderResource())
        r.putChild(b"nocontenttype", EmptyContentTypeHeaderResource())
        r.putChild(b"largechunkedfile", LargeChunkedFileResource())
        r.putChild(b"echo", Echo())
        self.site = server.Site(r, timeout=None)
        self.wrapper = WrappingFactory(self.site)
        self.host = 'localhost'
        if self.scheme == 'https':
            self.port = reactor.listenSSL(
                0, self.wrapper, ssl_context_factory(self.keyfile, self.certfile),
                interface=self.host)
        else:
            self.port = reactor.listenTCP(0, self.wrapper, interface=self.host)
        self.portno = self.port.getHost().port
        self.download_handler = self.download_handler_cls(Settings())
        self.download_request = self.download_handler.download_request

    @defer.inlineCallbacks
    def tearDown(self):
        yield self.port.stopListening()
        if hasattr(self.download_handler, 'close'):
            yield self.download_handler.close()
        shutil.rmtree(self.tmpname)

    def getURL(self, path):
        return "%s://%s:%d/%s" % (self.scheme, self.host, self.portno, path)

    def test_download(self):
        request = Request(self.getURL('file'))
        d = self.download_request(request, Spider('foo'))
        d.addCallback(lambda r: r.body)
        d.addCallback(self.assertEqual, b"0123456789")
        return d

    def test_download_head(self):
        request = Request(self.getURL('file'), method='HEAD')
        d = self.download_request(request, Spider('foo'))
        d.addCallback(lambda r: r.body)
        d.addCallback(self.assertEqual, b'')
        return d

    def test_redirect_status(self):
        request = Request(self.getURL('redirect'))
        d = self.download_request(request, Spider('foo'))
        d.addCallback(lambda r: r.status)
        d.addCallback(self.assertEqual, 302)
        return d

    def test_redirect_status_head(self):
        request = Request(self.getURL('redirect'), method='HEAD')
        d = self.download_request(request, Spider('foo'))
        d.addCallback(lambda r: r.status)
        d.addCallback(self.assertEqual, 302)
        return d

    @defer.inlineCallbacks
    def test_timeout_download_from_spider_nodata_rcvd(self):
        # client connects but no data is received
        spider = Spider('foo')
        meta = {'download_timeout': 0.2}
        request = Request(self.getURL('wait'), meta=meta)
        d = self.download_request(request, spider)
        yield self.assertFailure(d, defer.TimeoutError, error.TimeoutError)

    @defer.inlineCallbacks
    def test_timeout_download_from_spider_server_hangs(self):
        # client connects, server send headers and some body bytes but hangs
        spider = Spider('foo')
        meta = {'download_timeout': 0.2}
        request = Request(self.getURL('hang-after-headers'), meta=meta)
        d = self.download_request(request, spider)
        yield self.assertFailure(d, defer.TimeoutError, error.TimeoutError)

    def test_host_header_not_in_request_headers(self):
        def _test(response):
            self.assertEqual(
                response.body, to_bytes('%s:%d' % (self.host, self.portno)))
            self.assertEqual(request.headers, {})

        request = Request(self.getURL('host'))
        return self.download_request(request, Spider('foo')).addCallback(_test)

    def test_host_header_seted_in_request_headers(self):
        def _test(response):
            self.assertEqual(response.body, b'example.com')
            self.assertEqual(request.headers.get('Host'), b'example.com')

        request = Request(self.getURL('host'), headers={'Host': 'example.com'})
        return self.download_request(request, Spider('foo')).addCallback(_test)

        d = self.download_request(request, Spider('foo'))
        d.addCallback(lambda r: r.body)
        d.addCallback(self.assertEqual, b'example.com')
        return d

    def test_content_length_zero_bodyless_post_request_headers(self):
        """Tests if "Content-Length: 0" is sent for bodyless POST requests.

        This is not strictly required by HTTP RFCs but can cause trouble
        for some web servers.
        See:
        https://github.com/scrapy/scrapy/issues/823
        https://issues.apache.org/jira/browse/TS-2902
        https://github.com/kennethreitz/requests/issues/405
        https://bugs.python.org/issue14721
        """
        def _test(response):
            self.assertEqual(response.body, b'0')

        request = Request(self.getURL('contentlength'), method='POST', headers={'Host': 'example.com'})
        return self.download_request(request, Spider('foo')).addCallback(_test)

    def test_content_length_zero_bodyless_post_only_one(self):
        def _test(response):
            import json
            headers = Headers(json.loads(response.text)['headers'])
            contentlengths = headers.getlist('Content-Length')
            self.assertEqual(len(contentlengths), 1)
            self.assertEqual(contentlengths, [b"0"])

        request = Request(self.getURL('echo'), method='POST')
        return self.download_request(request, Spider('foo')).addCallback(_test)

    def test_payload(self):
        body = b'1'*100 # PayloadResource requires body length to be 100
        request = Request(self.getURL('payload'), method='POST', body=body)
        d = self.download_request(request, Spider('foo'))
        d.addCallback(lambda r: r.body)
        d.addCallback(self.assertEqual, body)
        return d


class MyPipeline(object):
    component_name = 'my_pipeline'

    def process_item(self, item, spider):
        logging.info('It Works!')
        return item
""")

        fname = abspath(join(self.proj_mod_path, 'settings.py'))
        with open(fname, 'a') as f:
            f.write("""
ITEM_PIPELINES = {'%s.pipelines.MyPipeline': 1}
""" % self.project_name)

    @defer.inlineCallbacks
    def test_spider_arguments(self):
        _, _, stderr = yield self.execute(['--spider', self.spider_name,
                                           '-a', 'test_arg=1',
                                           '-c', 'parse',
                                           self.url('/html')])
        self.assertIn("DEBUG: It Works!", to_native_str(stderr))

    @defer.inlineCallbacks
    def test_pipelines(self):
        _, _, stderr = yield self.execute(['--spider', self.spider_name,
                                           '--pipelines',
                                           '-c', 'parse',
                                           self.url('/html')])
        self.assertIn("INFO: It Works!", to_native_str(stderr))

    @defer.inlineCallbacks
    def test_parse_items(self):
        status, out, stderr = yield self.execute(
            ['--spider', self.spider_name, '-c', 'parse', self.url('/html')]
        )
        self.assertIn("""[{}, {'foo': 'bar'}]""", to_native_str(out))

    @defer.inlineCallbacks
    def test_parse_items_no_callback_passed(self):
        status, out, stderr = yield self.execute(
            ['--spider', self.spider_name, self.url('/html')]
        )
        self.assertIn("""[{}, {'foo': 'bar'}]""", to_native_str(out))

    @defer.inlineCallbacks
    def test_wrong_callback_passed(self):
        status, out, stderr = yield self.execute(
            ['--spider', self.spider_name, '-c', 'dummy', self.url('/html')]
        )
        self.assertRegexpMatches(to_native_str(out), """# Scraped Items  -+\n\[\]""")
        self.assertIn("""Cannot find callback""", to_native_str(stderr))

    @defer.inlineCallbacks
    def test_crawlspider_matching_rule_callback_set(self):
        """If a rule matches the URL, use it's defined callback."""
        status, out, stderr = yield self.execute(
            ['--spider', 'goodcrawl'+self.spider_name, '-r', self.url('/html')]
        )
        self.assertIn("""[{}, {'foo': 'bar'}]""", to_native_str(out))

    @defer.inlineCallbacks
    def test_crawlspider_matching_rule_default_callback(self):
        """If a rule match but it has no callback set, use the 'parse' callback."""
        status, out, stderr = yield self.execute(
            ['--spider', 'goodcrawl'+self.spider_name, '-r', self.url('/text')]
        )
        self.assertIn("""[{}, {'nomatch': 'default'}]""", to_native_str(out))

    @defer.inlineCallbacks
    def test_spider_with_no_rules_attribute(self):
        """Using -r with a spider with no rule should not produce items."""
        status, out, stderr = yield self.execute(
            ['--spider', self.spider_name, '-r', self.url('/html')]
        )
        self.assertRegexpMatches(to_native_str(out), """# Scraped Items  -+\n\[\]""")
        self.assertIn("""No CrawlSpider rules found""", to_native_str(stderr))

    @defer.inlineCallbacks
    def test_crawlspider_missing_callback(self):
        status, out, stderr = yield self.execute(
            ['--spider', 'badcrawl'+self.spider_name, '-r', self.url('/html')]
        )
        self.assertRegexpMatches(to_native_str(out), """# Scraped Items  -+\n\[\]""")

    @defer.inlineCallbacks
    def test_crawlspider_no_matching_rule(self):
        """The requested URL has no matching rule, so no items should be scraped"""
        status, out, stderr = yield self.execute(
            ['--spider', 'badcrawl'+self.spider_name, '-r', self.url('/enc-gb18030')]
        )
        self.assertRegexpMatches(to_native_str(out), """# Scraped Items  -+\n\[\]""")
        self.assertIn("""Cannot find a rule that matches""", to_native_str(stderr))

class TestHelper(unittest.TestCase):
    bbody = b'utf8-body'
    ubody = bbody.decode('utf8')
    txtresponse = TextResponse(url='http://example.org/', body=bbody, encoding='utf-8')
    response = Response(url='http://example.org/', body=bbody)

    def test_body_or_str(self):
        for obj in (self.bbody, self.ubody, self.txtresponse, self.response):
            r1 = _body_or_str(obj)
            self._assert_type_and_value(r1, self.ubody, obj)
            r2 = _body_or_str(obj, unicode=True)
            self._assert_type_and_value(r2, self.ubody, obj)
            r3 = _body_or_str(obj, unicode=False)
            self._assert_type_and_value(r3, self.bbody, obj)
            self.assertTrue(type(r1) is type(r2))
            self.assertTrue(type(r1) is not type(r3))


    def _assert_type_and_value(self, a, b, obj):
        self.assertTrue(type(a) is type(b),
                        'Got {}, expected {} for {!r}'.format(type(a), type(b), obj))
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()

class TestSettingsStrictOriginWhenCrossOrigin(MixinStrictOriginWhenCrossOrigin, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.StrictOriginWhenCrossOriginPolicy'}


class ShellTest(ProcessTest, SiteTest, unittest.TestCase):

    command = 'shell'

    @defer.inlineCallbacks
    def test_empty(self):
        _, out, _ = yield self.execute(['-c', 'item'])
        assert b'{}' in out

    @defer.inlineCallbacks
    def test_response_body(self):
        _, out, _ = yield self.execute([self.url('/text'), '-c', 'response.body'])
        assert b'Works' in out

    @defer.inlineCallbacks
    def test_response_type_text(self):
        _, out, _ = yield self.execute([self.url('/text'), '-c', 'type(response)'])
        assert b'TextResponse' in out

    @defer.inlineCallbacks
    def test_response_type_html(self):
        _, out, _ = yield self.execute([self.url('/html'), '-c', 'type(response)'])
        assert b'HtmlResponse' in out

    @defer.inlineCallbacks
    def test_response_selector_html(self):
        xpath = 'response.xpath("//p[@class=\'one\']/text()").extract()[0]'
        _, out, _ = yield self.execute([self.url('/html'), '-c', xpath])
        self.assertEqual(out.strip(), b'Works')

    @defer.inlineCallbacks
    def test_response_encoding_gb18030(self):
        _, out, _ = yield self.execute([self.url('/enc-gb18030'), '-c', 'response.encoding'])
        self.assertEqual(out.strip(), b'gb18030')

    @defer.inlineCallbacks
    def test_redirect(self):
        _, out, _ = yield self.execute([self.url('/redirect'), '-c', 'response.url'])
        assert out.strip().endswith(b'/redirected')

    @defer.inlineCallbacks
    def test_redirect_follow_302(self):
        _, out, _ = yield self.execute([self.url('/redirect-no-meta-refresh'), '-c', 'response.status'])
        assert out.strip().endswith(b'200')

    @defer.inlineCallbacks
    def test_redirect_not_follow_302(self):
        _, out, _ = yield self.execute(['--no-redirect', self.url('/redirect-no-meta-refresh'), '-c', 'response.status'])
        assert out.strip().endswith(b'302')

    @defer.inlineCallbacks
    def test_fetch_redirect_follow_302(self):
        """Test that calling `fetch(url)` follows HTTP redirects by default."""
        url = self.url('/redirect-no-meta-refresh')
        code = "fetch('{0}')"
        errcode, out, errout = yield self.execute(['-c', code.format(url)])
        self.assertEqual(errcode, 0, out)
        assert b'Redirecting (302)' in errout
        assert b'Crawled (200)' in errout

    @defer.inlineCallbacks
    def test_fetch_redirect_not_follow_302(self):
        """Test that calling `fetch(url, redirect=False)` disables automatic redirects."""
        url = self.url('/redirect-no-meta-refresh')
        code = "fetch('{0}', redirect=False)"
        errcode, out, errout = yield self.execute(['-c', code.format(url)])
        self.assertEqual(errcode, 0, out)
        assert b'Crawled (302)' in errout

    @defer.inlineCallbacks
    def test_request_replace(self):
        url = self.url('/text')
        code = "fetch('{0}') or fetch(response.request.replace(method='POST'))"
        errcode, out, _ = yield self.execute(['-c', code.format(url)])
        self.assertEqual(errcode, 0, out)

    @defer.inlineCallbacks
    def test_scrapy_import(self):
        url = self.url('/text')
        code = "fetch(scrapy.Request('{0}'))"
        errcode, out, _ = yield self.execute(['-c', code.format(url)])
        self.assertEqual(errcode, 0, out)

    @defer.inlineCallbacks
    def test_local_file(self):
        filepath = join(tests_datadir, 'test_site/index.html')
        _, out, _ = yield self.execute([filepath, '-c', 'item'])
        assert b'{}' in out

    @defer.inlineCallbacks
    def test_local_nofile(self):
        filepath = 'file:///tests/sample_data/test_site/nothinghere.html'
        errcode, out, err = yield self.execute([filepath, '-c', 'item'],
                                       check_code=False)
        self.assertEqual(errcode, 1, out or err)
        self.assertIn(b'No such file or directory', err)

    @defer.inlineCallbacks
    def test_dns_failures(self):
        url = 'www.somedomainthatdoesntexi.st'
        errcode, out, err = yield self.execute([url, '-c', 'item'],
                                       check_code=False)
        self.assertEqual(errcode, 1, out or err)
        self.assertIn(b'DNS lookup failed', err)

def _mocked_download_func(request, info):
    response = request.meta.get('response')
    return response() if callable(response) else response


class PythonItemExporterTest(BaseItemExporterTest):
    def _get_exporter(self, **kwargs):
        return PythonItemExporter(binary=False, **kwargs)

    def test_invalid_option(self):
        with self.assertRaisesRegexp(TypeError, "Unexpected options: invalid_option"):
            PythonItemExporter(invalid_option='something')

    def test_nested_item(self):
        i1 = TestItem(name=u'Joseph', age='22')
        i2 = dict(name=u'Maria', age=i1)
        i3 = TestItem(name=u'Jesus', age=i2)
        ie = self._get_exporter()
        exported = ie.export_item(i3)
        self.assertEqual(type(exported), dict)
        self.assertEqual(exported, {'age': {'age': {'age': '22', 'name': u'Joseph'}, 'name': u'Maria'}, 'name': 'Jesus'})
        self.assertEqual(type(exported['age']), dict)
        self.assertEqual(type(exported['age']['age']), dict)

    def test_export_list(self):
        i1 = TestItem(name=u'Joseph', age='22')
        i2 = TestItem(name=u'Maria', age=[i1])
        i3 = TestItem(name=u'Jesus', age=[i2])
        ie = self._get_exporter()
        exported = ie.export_item(i3)
        self.assertEqual(exported, {'age': [{'age': [{'age': '22', 'name': u'Joseph'}], 'name': u'Maria'}], 'name': 'Jesus'})
        self.assertEqual(type(exported['age'][0]), dict)
        self.assertEqual(type(exported['age'][0]['age'][0]), dict)

    def test_export_item_dict_list(self):
        i1 = TestItem(name=u'Joseph', age='22')
        i2 = dict(name=u'Maria', age=[i1])
        i3 = TestItem(name=u'Jesus', age=[i2])
        ie = self._get_exporter()
        exported = ie.export_item(i3)
        self.assertEqual(exported, {'age': [{'age': [{'age': '22', 'name': u'Joseph'}], 'name': u'Maria'}], 'name': 'Jesus'})
        self.assertEqual(type(exported['age'][0]), dict)
        self.assertEqual(type(exported['age'][0]['age'][0]), dict)

    def test_export_binary(self):
        exporter = PythonItemExporter(binary=True)
        value = TestItem(name=u'John\xa3', age=u'22')
        expected = {b'name': b'John\xc2\xa3', b'age': b'22'}
        self.assertEqual(expected, exporter.export_item(value))

    def test_nonstring_types_item(self):
        item = self._get_nonstring_types_item()
        ie = self._get_exporter()
        exported = ie.export_item(item)
        self.assertEqual(exported, item)


class TestHttpErrorMiddlewareHandleAll(TestCase):

    def setUp(self):
        self.spider = Spider('foo')
        self.mw = HttpErrorMiddleware(Settings({'HTTPERROR_ALLOW_ALL': True}))
        self.req = Request('http://scrapytest.org')
        self.res200, self.res404, self.res402 = _responses(self.req, [200, 404, 402])

    def test_process_spider_input(self):
        self.assertEqual(None,
                self.mw.process_spider_input(self.res200, self.spider))
        self.assertEqual(None,
                self.mw.process_spider_input(self.res404, self.spider))

    def test_meta_overrides_settings(self):
        request = Request('http://scrapytest.org',
                              meta={'handle_httpstatus_list': [404]})
        res404 = self.res404.copy()
        res404.request = request
        res402 = self.res402.copy()
        res402.request = request

        self.assertEqual(None,
            self.mw.process_spider_input(res404, self.spider))
        self.assertRaises(HttpError,
                self.mw.process_spider_input, res402, self.spider)


def _mocked_download_func(request, info):
    response = request.meta.get('response')
    return response() if callable(response) else response


class Https11TestCase(Http11TestCase):
    scheme = 'https'


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


class FilesystemStorageGzipTest(FilesystemStorageTest):

    def _get_settings(self, **new_settings):
        new_settings.setdefault('HTTPCACHE_GZIP', True)
        return super(FilesystemStorageTest, self)._get_settings(**new_settings)

class XmliterTestCase(unittest.TestCase):

    xmliter = staticmethod(xmliter)

    def test_xmliter(self):
        body = b"""<?xml version="1.0" encoding="UTF-8"?>\
            <products xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="someschmea.xsd">\
              <product id="001">\
                <type>Type 1</type>\
                <name>Name 1</name>\
              </product>\
              <product id="002">\
                <type>Type 2</type>\
                <name>Name 2</name>\
              </product>\
            </products>"""

        response = XmlResponse(url="http://example.com", body=body)
        attrs = []
        for x in self.xmliter(response, 'product'):
            attrs.append((x.xpath("@id").extract(), x.xpath("name/text()").extract(), x.xpath("./type/text()").extract()))

        self.assertEqual(attrs,
                         [(['001'], ['Name 1'], ['Type 1']), (['002'], ['Name 2'], ['Type 2'])])

    def test_xmliter_unusual_node(self):
        body = b"""<?xml version="1.0" encoding="UTF-8"?>
            <root>
                <matchme...></matchme...>
                <matchmenot></matchmenot>
            </root>
        """
        response = XmlResponse(url="http://example.com", body=body)
        nodenames = [e.xpath('name()').extract()
                 for e in self.xmliter(response, 'matchme...')]
        self.assertEqual(nodenames, [['matchme...']])

    def test_xmliter_unicode(self):
        # example taken from https://github.com/scrapy/scrapy/issues/1665
        body = u"""<?xml version="1.0" encoding="UTF-8"?>
            <þingflokkar>
               <þingflokkur id="26">
                  <heiti />
                  <skammstafanir>
                     <stuttskammstöfun>-</stuttskammstöfun>
                     <löngskammstöfun />
                  </skammstafanir>
                  <tímabil>
                     <fyrstaþing>80</fyrstaþing>
                  </tímabil>
               </þingflokkur>
               <þingflokkur id="21">
                  <heiti>Alþýðubandalag</heiti>
                  <skammstafanir>
                     <stuttskammstöfun>Ab</stuttskammstöfun>
                     <löngskammstöfun>Alþb.</löngskammstöfun>
                  </skammstafanir>
                  <tímabil>
                     <fyrstaþing>76</fyrstaþing>
                     <síðastaþing>123</síðastaþing>
                  </tímabil>
               </þingflokkur>
               <þingflokkur id="27">
                  <heiti>Alþýðuflokkur</heiti>
                  <skammstafanir>
                     <stuttskammstöfun>A</stuttskammstöfun>
                     <löngskammstöfun>Alþfl.</löngskammstöfun>
                  </skammstafanir>
                  <tímabil>
                     <fyrstaþing>27</fyrstaþing>
                     <síðastaþing>120</síðastaþing>
                  </tímabil>
               </þingflokkur>
            </þingflokkar>"""

        for r in (
            # with bytes
            XmlResponse(url="http://example.com", body=body.encode('utf-8')),
            # Unicode body needs encoding information
            XmlResponse(url="http://example.com", body=body, encoding='utf-8')):

            attrs = []
            for x in self.xmliter(r, u'þingflokkur'):
                attrs.append((x.xpath('@id').extract(),
                              x.xpath(u'./skammstafanir/stuttskammstöfun/text()').extract(),
                              x.xpath(u'./tímabil/fyrstaþing/text()').extract()))

            self.assertEqual(attrs,
                             [([u'26'], [u'-'], [u'80']),
                              ([u'21'], [u'Ab'], [u'76']),
                              ([u'27'], [u'A'], [u'27'])])

    def test_xmliter_text(self):
        body = u"""<?xml version="1.0" encoding="UTF-8"?><products><product>one</product><product>two</product></products>"""

        self.assertEqual([x.xpath("text()").extract() for x in self.xmliter(body, 'product')],
                         [[u'one'], [u'two']])

    def test_xmliter_namespaces(self):
        body = b"""\
            <?xml version="1.0" encoding="UTF-8"?>
            <rss version="2.0" xmlns:g="http://base.google.com/ns/1.0">
                <channel>
                <title>My Dummy Company</title>
                <link>http://www.mydummycompany.com</link>
                <description>This is a dummy company. We do nothing.</description>
                <item>
                    <title>Item 1</title>
                    <description>This is item 1</description>
                    <link>http://www.mydummycompany.com/items/1</link>
                    <g:image_link>http://www.mydummycompany.com/images/item1.jpg</g:image_link>
                    <g:id>ITEM_1</g:id>
                    <g:price>400</g:price>
                </item>
                </channel>
            </rss>
        """
        response = XmlResponse(url='http://mydummycompany.com', body=body)
        my_iter = self.xmliter(response, 'item')

        node = next(my_iter)
        node.register_namespace('g', 'http://base.google.com/ns/1.0')
        self.assertEqual(node.xpath('title/text()').extract(), ['Item 1'])
        self.assertEqual(node.xpath('description/text()').extract(), ['This is item 1'])
        self.assertEqual(node.xpath('link/text()').extract(), ['http://www.mydummycompany.com/items/1'])
        self.assertEqual(node.xpath('g:image_link/text()').extract(), ['http://www.mydummycompany.com/images/item1.jpg'])
        self.assertEqual(node.xpath('g:id/text()').extract(), ['ITEM_1'])
        self.assertEqual(node.xpath('g:price/text()').extract(), ['400'])
        self.assertEqual(node.xpath('image_link/text()').extract(), [])
        self.assertEqual(node.xpath('id/text()').extract(), [])
        self.assertEqual(node.xpath('price/text()').extract(), [])

    def test_xmliter_exception(self):
        body = u"""<?xml version="1.0" encoding="UTF-8"?><products><product>one</product><product>two</product></products>"""

        iter = self.xmliter(body, 'product')
        next(iter)
        next(iter)

        self.assertRaises(StopIteration, next, iter)

    def test_xmliter_objtype_exception(self):
        i = self.xmliter(42, 'product')
        self.assertRaises(AssertionError, next, i)

    def test_xmliter_encoding(self):
        body = b'<?xml version="1.0" encoding="ISO-8859-9"?>\n<xml>\n    <item>Some Turkish Characters \xd6\xc7\xde\xdd\xd0\xdc \xfc\xf0\xfd\xfe\xe7\xf6</item>\n</xml>\n\n'
        response = XmlResponse('http://www.example.com', body=body)
        self.assertEqual(
            next(self.xmliter(response, 'item')).extract(),
            u'<item>Some Turkish Characters \xd6\xc7\u015e\u0130\u011e\xdc \xfc\u011f\u0131\u015f\xe7\xf6</item>'
        )


class MySpider(scrapy.Spider):
    name = 'myspider'
    start_urls = ['http://localhost:12345']

    def parse(self, response):
        return {'test': 'value'}
""",
                           args=('-s', 'DNSCACHE_ENABLED=False'))
        print(log)
        self.assertNotIn("DNSLookupError", log)
        self.assertIn("INFO: Spider opened", log)

    def test_runspider_log_short_names(self):
        log1 = self.get_log(self.debug_log_spider,
                            args=('-s', 'LOG_SHORT_NAMES=1'))
        print(log1)
        self.assertIn("[myspider] DEBUG: It Works!", log1)
        self.assertIn("[scrapy]", log1)
        self.assertNotIn("[scrapy.core.engine]", log1)

        log2 = self.get_log(self.debug_log_spider,
                            args=('-s', 'LOG_SHORT_NAMES=0'))
        print(log2)
        self.assertIn("[myspider] DEBUG: It Works!", log2)
        self.assertNotIn("[scrapy]", log2)
        self.assertIn("[scrapy.core.engine]", log2)

    def test_runspider_no_spider_found(self):
        log = self.get_log("from scrapy.spiders import Spider\n")
        self.assertIn("No spider found in file", log)

    def test_runspider_file_not_found(self):
        p = self.proc('runspider', 'some_non_existent_file')
        log = to_native_str(p.stderr.read())
        self.assertIn("File not found: some_non_existent_file", log)

    def test_runspider_unable_to_load(self):
        log = self.get_log('', name='myspider.txt')
        self.assertIn('Unable to load', log)

    def test_start_requests_errors(self):
        log = self.get_log("""
import scrapy

class MetaRefreshMiddlewareTest(unittest.TestCase):

    def setUp(self):
        crawler = get_crawler(Spider)
        self.spider = crawler._create_spider('foo')
        self.mw = MetaRefreshMiddleware.from_crawler(crawler)

    def _body(self, interval=5, url='http://example.org/newpage'):
        html = u"""<html><head><meta http-equiv="refresh" content="{0};url={1}"/></head></html>"""
        return html.format(interval, url).encode('utf-8')

    def test_priority_adjust(self):
        req = Request('http://a.com')
        rsp = HtmlResponse(req.url, body=self._body())
        req2 = self.mw.process_response(req, rsp, self.spider)
        assert req2.priority > req.priority

    def test_meta_refresh(self):
        req = Request(url='http://example.org')
        rsp = HtmlResponse(req.url, body=self._body())
        req2 = self.mw.process_response(req, rsp, self.spider)
        assert isinstance(req2, Request)
        self.assertEqual(req2.url, 'http://example.org/newpage')

    def test_meta_refresh_with_high_interval(self):
        # meta-refresh with high intervals don't trigger redirects
        req = Request(url='http://example.org')
        rsp = HtmlResponse(url='http://example.org',
                           body=self._body(interval=1000),
                           encoding='utf-8')
        rsp2 = self.mw.process_response(req, rsp, self.spider)
        assert rsp is rsp2

    def test_meta_refresh_trough_posted_request(self):
        req = Request(url='http://example.org', method='POST', body='test',
                      headers={'Content-Type': 'text/plain', 'Content-length': '4'})
        rsp = HtmlResponse(req.url, body=self._body())
        req2 = self.mw.process_response(req, rsp, self.spider)

        assert isinstance(req2, Request)
        self.assertEqual(req2.url, 'http://example.org/newpage')
        self.assertEqual(req2.method, 'GET')
        assert 'Content-Type' not in req2.headers, \
            "Content-Type header must not be present in redirected request"
        assert 'Content-Length' not in req2.headers, \
            "Content-Length header must not be present in redirected request"
        assert not req2.body, \
            "Redirected body must be empty, not '%s'" % req2.body

    def test_max_redirect_times(self):
        self.mw.max_redirect_times = 1
        req = Request('http://scrapytest.org/max')
        rsp = HtmlResponse(req.url, body=self._body())

        req = self.mw.process_response(req, rsp, self.spider)
        assert isinstance(req, Request)
        assert 'redirect_times' in req.meta
        self.assertEqual(req.meta['redirect_times'], 1)
        self.assertRaises(IgnoreRequest, self.mw.process_response, req, rsp, self.spider)

    def test_ttl(self):
        self.mw.max_redirect_times = 100
        req = Request('http://scrapytest.org/302', meta={'redirect_ttl': 1})
        rsp = HtmlResponse(req.url, body=self._body())

        req = self.mw.process_response(req, rsp, self.spider)
        assert isinstance(req, Request)
        self.assertRaises(IgnoreRequest, self.mw.process_response, req, rsp, self.spider)

    def test_redirect_urls(self):
        req1 = Request('http://scrapytest.org/first')
        rsp1 = HtmlResponse(req1.url, body=self._body(url='/redirected'))
        req2 = self.mw.process_response(req1, rsp1, self.spider)
        assert isinstance(req2, Request), req2
        rsp2 = HtmlResponse(req2.url, body=self._body(url='/redirected2'))
        req3 = self.mw.process_response(req2, rsp2, self.spider)
        assert isinstance(req3, Request), req3
        self.assertEqual(req2.url, 'http://scrapytest.org/redirected')
        self.assertEqual(req2.meta['redirect_urls'], ['http://scrapytest.org/first'])
        self.assertEqual(req3.url, 'http://scrapytest.org/redirected2')
        self.assertEqual(req3.meta['redirect_urls'], ['http://scrapytest.org/first', 'http://scrapytest.org/redirected'])


if __name__ == "__main__":
    unittest.main()
