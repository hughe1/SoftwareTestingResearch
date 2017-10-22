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

class FTPTestCase(BaseFTPTestCase):

    def test_invalid_credentials(self):
        from twisted.protocols.ftp import ConnectionLost

        meta = dict(self.req_meta)
        meta.update({"ftp_password": 'invalid'})
        request = Request(url="ftp://127.0.0.1:%s/file.txt" % self.portNum,
                          meta=meta)
        d = self.download_handler.download_request(request, None)

        def _test(r):
            self.assertEqual(r.type, ConnectionLost)
        return self._add_test_callbacks(d, errback=_test)


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


class LeveldbStorageTest(DefaultStorageTest):

    pytest.importorskip('leveldb')
    storage_class = 'scrapy.extensions.httpcache.LeveldbCacheStorage'


class MyBadCrawlSpider(CrawlSpider):
    '''Spider which doesn't define a parse_item callback while using it in a rule.'''
    name = 'badcrawl{0}'

    rules = (
        Rule(LinkExtractor(allow=r'/html'), callback='parse_item', follow=True),
    )

    def parse(self, response):
        return [scrapy.Item(), dict(foo='bar')]
""".format(self.spider_name))

        fname = abspath(join(self.proj_mod_path, 'pipelines.py'))
        with open(fname, 'w') as f:
            f.write("""
import logging

class AjaxCrawlMiddlewareTest(unittest.TestCase):
    def setUp(self):
        crawler = get_crawler(Spider, {'AJAXCRAWL_ENABLED': True})
        self.spider = crawler._create_spider('foo')
        self.mw = AjaxCrawlMiddleware.from_crawler(crawler)

    def _ajaxcrawlable_body(self):
        return b'<html><head><meta name="fragment" content="!"/></head><body></body></html>'

    def _req_resp(self, url, req_kwargs=None, resp_kwargs=None):
        req = Request(url, **(req_kwargs or {}))
        resp = HtmlResponse(url, request=req, **(resp_kwargs or {}))
        return req, resp

    def test_non_get(self):
        req, resp = self._req_resp('http://example.com/', {'method': 'HEAD'})
        resp2 = self.mw.process_response(req, resp, self.spider)
        self.assertEqual(resp, resp2)

    def test_binary_response(self):
        req = Request('http://example.com/')
        resp = Response('http://example.com/', body=b'foobar\x00\x01\x02', request=req)
        resp2 = self.mw.process_response(req, resp, self.spider)
        self.assertIs(resp, resp2)

    def test_ajaxcrawl(self):
        req, resp = self._req_resp(
            'http://example.com/',
            {'meta': {'foo': 'bar'}},
            {'body': self._ajaxcrawlable_body()}
        )
        req2 = self.mw.process_response(req, resp, self.spider)
        self.assertEqual(req2.url, 'http://example.com/?_escaped_fragment_=')
        self.assertEqual(req2.meta['foo'], 'bar')

    def test_ajaxcrawl_loop(self):
        req, resp = self._req_resp('http://example.com/', {}, {'body': self._ajaxcrawlable_body()})
        req2 = self.mw.process_response(req, resp, self.spider)
        resp2 = HtmlResponse(req2.url, body=resp.body, request=req2)
        resp3 = self.mw.process_response(req2, resp2, self.spider)

        assert isinstance(resp3, HtmlResponse), (resp3.__class__, resp3)
        self.assertEqual(resp3.request.url, 'http://example.com/?_escaped_fragment_=')
        assert resp3 is resp2

    def test_noncrawlable_body(self):
        req, resp = self._req_resp('http://example.com/', {}, {'body': b'<html></html>'})
        resp2 = self.mw.process_response(req, resp, self.spider)
        self.assertIs(resp, resp2)

class HeadersTest(unittest.TestCase):

    def assertSortedEqual(self, first, second, msg=None):
        return self.assertEqual(sorted(first), sorted(second), msg)

    def test_basics(self):
        h = Headers({'Content-Type': 'text/html', 'Content-Length': 1234})
        assert h['Content-Type']
        assert h['Content-Length']

        self.assertRaises(KeyError, h.__getitem__, 'Accept')
        self.assertEqual(h.get('Accept'), None)
        self.assertEqual(h.getlist('Accept'), [])

        self.assertEqual(h.get('Accept', '*/*'), b'*/*')
        self.assertEqual(h.getlist('Accept', '*/*'), [b'*/*'])
        self.assertEqual(h.getlist('Accept', ['text/html', 'images/jpeg']),
                         [b'text/html', b'images/jpeg'])

    def test_single_value(self):
        h = Headers()
        h['Content-Type'] = 'text/html'
        self.assertEqual(h['Content-Type'], b'text/html')
        self.assertEqual(h.get('Content-Type'), b'text/html')
        self.assertEqual(h.getlist('Content-Type'), [b'text/html'])

    def test_multivalue(self):
        h = Headers()
        h['X-Forwarded-For'] = hlist = ['ip1', 'ip2']
        self.assertEqual(h['X-Forwarded-For'], b'ip2')
        self.assertEqual(h.get('X-Forwarded-For'), b'ip2')
        self.assertEqual(h.getlist('X-Forwarded-For'), [b'ip1', b'ip2'])
        assert h.getlist('X-Forwarded-For') is not hlist

    def test_encode_utf8(self):
        h = Headers({u'key': u'\xa3'}, encoding='utf-8')
        key, val = dict(h).popitem()
        assert isinstance(key, bytes), key
        assert isinstance(val[0], bytes), val[0]
        self.assertEqual(val[0], b'\xc2\xa3')

    def test_encode_latin1(self):
        h = Headers({u'key': u'\xa3'}, encoding='latin1')
        key, val = dict(h).popitem()
        self.assertEqual(val[0], b'\xa3')

    def test_encode_multiple(self):
        h = Headers({u'key': [u'\xa3']}, encoding='utf-8')
        key, val = dict(h).popitem()
        self.assertEqual(val[0], b'\xc2\xa3')

    def test_delete_and_contains(self):
        h = Headers()
        h['Content-Type'] = 'text/html'
        assert 'Content-Type' in h
        del h['Content-Type']
        assert 'Content-Type' not in h

    def test_setdefault(self):
        h = Headers()
        hlist = ['ip1', 'ip2']
        olist = h.setdefault('X-Forwarded-For', hlist)
        assert h.getlist('X-Forwarded-For') is not hlist
        assert h.getlist('X-Forwarded-For') is olist

        h = Headers()
        olist = h.setdefault('X-Forwarded-For', 'ip1')
        self.assertEqual(h.getlist('X-Forwarded-For'), [b'ip1'])
        assert h.getlist('X-Forwarded-For') is olist

    def test_iterables(self):
        idict = {'Content-Type': 'text/html', 'X-Forwarded-For': ['ip1', 'ip2']}

        h = Headers(idict)
        self.assertDictEqual(dict(h),
                             {b'Content-Type': [b'text/html'],
                              b'X-Forwarded-For': [b'ip1', b'ip2']})
        self.assertSortedEqual(h.keys(),
                               [b'X-Forwarded-For', b'Content-Type'])
        self.assertSortedEqual(h.items(),
                               [(b'X-Forwarded-For', [b'ip1', b'ip2']),
                                (b'Content-Type', [b'text/html'])])
        self.assertSortedEqual(h.iteritems(),
                               [(b'X-Forwarded-For', [b'ip1', b'ip2']),
                                (b'Content-Type', [b'text/html'])])
        self.assertSortedEqual(h.values(), [b'ip2', b'text/html'])

    def test_update(self):
        h = Headers()
        h.update({'Content-Type': 'text/html',
                  'X-Forwarded-For': ['ip1', 'ip2']})
        self.assertEqual(h.getlist('Content-Type'), [b'text/html'])
        self.assertEqual(h.getlist('X-Forwarded-For'), [b'ip1', b'ip2'])

    def test_copy(self):
        h1 = Headers({'header1': ['value1', 'value2']})
        h2 = copy.copy(h1)
        self.assertEqual(h1, h2)
        self.assertEqual(h1.getlist('header1'), h2.getlist('header1'))
        assert h1.getlist('header1') is not h2.getlist('header1')
        assert isinstance(h2, Headers)

    def test_appendlist(self):
        h1 = Headers({'header1': 'value1'})
        h1.appendlist('header1', 'value3')
        self.assertEqual(h1.getlist('header1'), [b'value1', b'value3'])

        h1 = Headers()
        h1.appendlist('header1', 'value1')
        h1.appendlist('header1', 'value3')
        self.assertEqual(h1.getlist('header1'), [b'value1', b'value3'])

    def test_setlist(self):
        h1 = Headers({'header1': 'value1'})
        self.assertEqual(h1.getlist('header1'), [b'value1'])
        h1.setlist('header1', [b'value2', b'value3'])
        self.assertEqual(h1.getlist('header1'), [b'value2', b'value3'])

    def test_setlistdefault(self):
        h1 = Headers({'header1': 'value1'})
        h1.setlistdefault('header1', ['value2', 'value3'])
        h1.setlistdefault('header2', ['value2', 'value3'])
        self.assertEqual(h1.getlist('header1'), [b'value1'])
        self.assertEqual(h1.getlist('header2'), [b'value2', b'value3'])

    def test_none_value(self):
        h1 = Headers()
        h1['foo'] = 'bar'
        h1['foo'] = None
        h1.setdefault('foo', 'bar')
        self.assertEqual(h1.get('foo'), None)
        self.assertEqual(h1.getlist('foo'), [])

    def test_int_value(self):
        h1 = Headers({'hey': 5})
        h1['foo'] = 1
        h1.setdefault('bar', 2)
        h1.setlist('buz', [1, 'dos', 3])
        self.assertEqual(h1.getlist('foo'), [b'1'])
        self.assertEqual(h1.getlist('bar'), [b'2'])
        self.assertEqual(h1.getlist('buz'), [b'1', b'dos', b'3'])
        self.assertEqual(h1.getlist('hey'), [b'5'])

    def test_invalid_value(self):
        self.assertRaisesRegexp(TypeError, 'Unsupported value type',
                                Headers, {'foo': object()})
        self.assertRaisesRegexp(TypeError, 'Unsupported value type',
                                Headers().__setitem__, 'foo', object())
        self.assertRaisesRegexp(TypeError, 'Unsupported value type',
                                Headers().setdefault, 'foo', object())
        self.assertRaisesRegexp(TypeError, 'Unsupported value type',
                                Headers().setlist, 'foo', [object()])

class DeferUtilsTest(unittest.TestCase):

    @defer.inlineCallbacks
    def test_process_chain(self):
        x = yield process_chain([cb1, cb2, cb3], 'res', 'v1', 'v2')
        self.assertEqual(x, "(cb3 (cb2 (cb1 res v1 v2) v1 v2) v1 v2)")

        gotexc = False
        try:
            yield process_chain([cb1, cb_fail, cb3], 'res', 'v1', 'v2')
        except TypeError as e:
            gotexc = True
        self.assertTrue(gotexc)

    @defer.inlineCallbacks
    def test_process_chain_both(self):
        x = yield process_chain_both([cb_fail, cb2, cb3], [None, eb1, None], 'res', 'v1', 'v2')
        self.assertEqual(x, "(cb3 (eb1 TypeError v1 v2) v1 v2)")

        fail = Failure(ZeroDivisionError())
        x = yield process_chain_both([eb1, cb2, cb3], [eb1, None, None], fail, 'v1', 'v2')
        self.assertEqual(x, "(cb3 (cb2 (eb1 ZeroDivisionError v1 v2) v1 v2) v1 v2)")

    @defer.inlineCallbacks
    def test_process_parallel(self):
        x = yield process_parallel([cb1, cb2, cb3], 'res', 'v1', 'v2')
        self.assertEqual(x, ['(cb1 res v1 v2)', '(cb2 res v1 v2)', '(cb3 res v1 v2)'])

    def test_process_parallel_failure(self):
        d = process_parallel([cb1, cb_fail, cb3], 'res', 'v1', 'v2')
        self.failUnlessFailure(d, TypeError)
        return d


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

class DeferUtilsTest(unittest.TestCase):

    @defer.inlineCallbacks
    def test_process_chain(self):
        x = yield process_chain([cb1, cb2, cb3], 'res', 'v1', 'v2')
        self.assertEqual(x, "(cb3 (cb2 (cb1 res v1 v2) v1 v2) v1 v2)")

        gotexc = False
        try:
            yield process_chain([cb1, cb_fail, cb3], 'res', 'v1', 'v2')
        except TypeError as e:
            gotexc = True
        self.assertTrue(gotexc)

    @defer.inlineCallbacks
    def test_process_chain_both(self):
        x = yield process_chain_both([cb_fail, cb2, cb3], [None, eb1, None], 'res', 'v1', 'v2')
        self.assertEqual(x, "(cb3 (eb1 TypeError v1 v2) v1 v2)")

        fail = Failure(ZeroDivisionError())
        x = yield process_chain_both([eb1, cb2, cb3], [eb1, None, None], fail, 'v1', 'v2')
        self.assertEqual(x, "(cb3 (cb2 (eb1 ZeroDivisionError v1 v2) v1 v2) v1 v2)")

    @defer.inlineCallbacks
    def test_process_parallel(self):
        x = yield process_parallel([cb1, cb2, cb3], 'res', 'v1', 'v2')
        self.assertEqual(x, ['(cb1 res v1 v2)', '(cb2 res v1 v2)', '(cb3 res v1 v2)'])

    def test_process_parallel_failure(self):
        d = process_parallel([cb1, cb_fail, cb3], 'res', 'v1', 'v2')
        self.failUnlessFailure(d, TypeError)
        return d


class Https11InvalidDNSId(Https11TestCase):
    """Connect to HTTPS hosts with IP while certificate uses domain names IDs."""

    def setUp(self):
        super(Https11InvalidDNSId, self).setUp()
        self.host = '127.0.0.1'


class DeprecationTest(unittest.TestCase):

    def test_basespider_is_deprecated(self):
        with warnings.catch_warnings(record=True) as w:

            class MySpider1(BaseSpider):
                pass

            self.assertEqual(len(w), 1)
            self.assertEqual(w[0].category, ScrapyDeprecationWarning)
            self.assertEqual(w[0].lineno, inspect.getsourcelines(MySpider1)[1])

    def test_basespider_issubclass(self):
        class MySpider2(Spider):
            pass

        class MySpider2a(MySpider2):
            pass

        class Foo(object):
            pass

        class Foo2(object_ref):
            pass

        assert issubclass(MySpider2, BaseSpider)
        assert issubclass(MySpider2a, BaseSpider)
        assert not issubclass(Foo, BaseSpider)
        assert not issubclass(Foo2, BaseSpider)

    def test_basespider_isinstance(self):
        class MySpider3(Spider):
            name = 'myspider3'

        class MySpider3a(MySpider3):
            pass

        class Foo(object):
            pass

        class Foo2(object_ref):
            pass

        assert isinstance(MySpider3(), BaseSpider)
        assert isinstance(MySpider3a(), BaseSpider)
        assert not isinstance(Foo(), BaseSpider)
        assert not isinstance(Foo2(), BaseSpider)

    def test_crawl_spider(self):
        assert issubclass(CrawlSpider, Spider)
        assert issubclass(CrawlSpider, BaseSpider)
        assert isinstance(CrawlSpider(name='foo'), Spider)
        assert isinstance(CrawlSpider(name='foo'), BaseSpider)

    def test_make_requests_from_url_deprecated(self):
        class MySpider4(Spider):
            name = 'spider1'
            start_urls = ['http://example.com']

        class MySpider5(Spider):
            name = 'spider2'
            start_urls = ['http://example.com']

            def make_requests_from_url(self, url):
                return Request(url + "/foo", dont_filter=True)

        with warnings.catch_warnings(record=True) as w:
            # spider without overridden make_requests_from_url method
            # doesn't issue a warning
            spider1 = MySpider4()
            self.assertEqual(len(list(spider1.start_requests())), 1)
            self.assertEqual(len(w), 0)

            # spider with overridden make_requests_from_url issues a warning,
            # but the method still works
            spider2 = MySpider5()
            requests = list(spider2.start_requests())
            self.assertEqual(len(requests), 1)
            self.assertEqual(requests[0].url, 'http://example.com/foo')
            self.assertEqual(len(w), 1)


class Https11TestCase(Http11TestCase):
    scheme = 'https'


class FileTestCase(unittest.TestCase):

    def setUp(self):
        self.tmpname = self.mktemp()
        with open(self.tmpname + '^', 'w') as f:
            f.write('0123456789')
        self.download_request = FileDownloadHandler(Settings()).download_request

    def tearDown(self):
        os.unlink(self.tmpname + '^')

    def test_download(self):
        def _test(response):
            self.assertEqual(response.url, request.url)
            self.assertEqual(response.status, 200)
            self.assertEqual(response.body, b'0123456789')

        request = Request(path_to_file_uri(self.tmpname + '^'))
        assert request.url.upper().endswith('%5E')
        return self.download_request(request, Spider('foo')).addCallback(_test)

    def test_non_existent(self):
        request = Request('file://%s' % self.mktemp())
        d = self.download_request(request, Spider('foo'))
        return self.assertFailure(d, IOError)


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

class ManagerTestCase(TestCase):

    settings_dict = None

    def setUp(self):
        self.crawler = get_crawler(Spider, self.settings_dict)
        self.spider = self.crawler._create_spider('foo')
        self.mwman = DownloaderMiddlewareManager.from_crawler(self.crawler)
        # some mw depends on stats collector
        self.crawler.stats.open_spider(self.spider)
        return self.mwman.open_spider(self.spider)

    def tearDown(self):
        self.crawler.stats.close_spider(self.spider, '')
        return self.mwman.close_spider(self.spider)

    def _download(self, request, response=None):
        """Executes downloader mw manager's download method and returns
        the result (Request or Response) or raise exception in case of
        failure.
        """
        if not response:
            response = Response(request.url)

        def download_func(**kwargs):
            return response

        dfd = self.mwman.download(download_func, request, self.spider)
        # catch deferred result and return the value
        results = []
        dfd.addBoth(results.append)
        self._wait(dfd)
        ret = results[0]
        if isinstance(ret, Failure):
            ret.raiseException()
        return ret


class ChunkSize1MarshalFifoDiskQueueTest(MarshalFifoDiskQueueTest):
    chunksize = 1

class TestSettingsOriginWhenCrossOrigin(MixinOriginWhenCrossOrigin, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.OriginWhenCrossOriginPolicy'}


class PickleFifoDiskQueueTest(MarshalFifoDiskQueueTest):

    chunksize = 100000

    def queue(self):
        return PickleFifoDiskQueue(self.qpath, chunksize=self.chunksize)

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

class CustomRequest(Request):
    pass

class ChunkSize2MarshalFifoDiskQueueTest(MarshalFifoDiskQueueTest):
    chunksize = 2

class ChunkSize2PickleFifoDiskQueueTest(PickleFifoDiskQueueTest):
    chunksize = 2

class ResponseFromProcessRequestTest(ManagerTestCase):
    """Tests middleware returning a response from process_request."""

    def test_download_func_not_called(self):
        resp = Response('http://example.com/index.html')

        class ResponseMiddleware(object):
            def process_request(self, request, spider):
                return resp

        self.mwman._add_middleware(ResponseMiddleware())

        req = Request('http://example.com/index.html')
        download_func = mock.MagicMock()
        dfd = self.mwman.download(download_func, req, self.spider)
        results = []
        dfd.addBoth(results.append)
        self._wait(dfd)

        self.assertIs(results[0], resp)
        self.assertFalse(download_func.called)

class SendCatchLogDeferredTest2(SendCatchLogTest):

    def ok_handler(self, arg, handlers_called):
        handlers_called.add(self.ok_handler)
        assert arg == 'test'
        d = defer.Deferred()
        reactor.callLater(0, d.callback, "OK")
        return d

    def _get_result(self, signal, *a, **kw):
        return send_catch_log_deferred(signal, *a, **kw)

class TestSpider(Spider):
    http_user = 'foo'
    http_pass = 'bar'


class DataURITestCase(unittest.TestCase):

    def setUp(self):
        self.download_handler = DataURIDownloadHandler(Settings())
        self.download_request = self.download_handler.download_request
        self.spider = Spider('foo')

    def test_response_attrs(self):
        uri = "data:,A%20brief%20note"

        def _test(response):
            self.assertEqual(response.url, uri)
            self.assertFalse(response.headers)

        request = Request(uri)
        return self.download_request(request, self.spider).addCallback(_test)

    def test_default_mediatype_encoding(self):
        def _test(response):
            self.assertEqual(response.text, 'A brief note')
            self.assertEqual(type(response),
                              responsetypes.from_mimetype("text/plain"))
            self.assertEqual(response.encoding, "US-ASCII")

        request = Request("data:,A%20brief%20note")
        return self.download_request(request, self.spider).addCallback(_test)

    def test_default_mediatype(self):
        def _test(response):
            self.assertEqual(response.text, u'\u038e\u03a3\u038e')
            self.assertEqual(type(response),
                              responsetypes.from_mimetype("text/plain"))
            self.assertEqual(response.encoding, "iso-8859-7")

        request = Request("data:;charset=iso-8859-7,%be%d3%be")
        return self.download_request(request, self.spider).addCallback(_test)

    def test_text_charset(self):
        def _test(response):
            self.assertEqual(response.text, u'\u038e\u03a3\u038e')
            self.assertEqual(response.body, b'\xbe\xd3\xbe')
            self.assertEqual(response.encoding, "iso-8859-7")

        request = Request("data:text/plain;charset=iso-8859-7,%be%d3%be")
        return self.download_request(request, self.spider).addCallback(_test)

    def test_mediatype_parameters(self):
        def _test(response):
            self.assertEqual(response.text, u'\u038e\u03a3\u038e')
            self.assertEqual(type(response),
                              responsetypes.from_mimetype("text/plain"))
            self.assertEqual(response.encoding, "utf-8")

        request = Request('data:text/plain;foo=%22foo;bar%5C%22%22;'
                          'charset=utf-8;bar=%22foo;%5C%22 foo ;/,%22'
                          ',%CE%8E%CE%A3%CE%8E')
        return self.download_request(request, self.spider).addCallback(_test)

    def test_base64(self):
        def _test(response):
            self.assertEqual(response.text, 'Hello, world.')

        request = Request('data:text/plain;base64,SGVsbG8sIHdvcmxkLg%3D%3D')
        return self.download_request(request, self.spider).addCallback(_test)

class MOff(object):

    def open_spider(self, spider):
        pass

    def close_spider(self, spider):
        pass

    def __init__(self):
        raise NotConfigured


class TestRequestMetaSettingFallback(TestCase):

    params = [
        (
            # When an unknown policy is referenced in Request.meta
            # (here, a typo error),
            # the policy defined in settings takes precedence
            {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.OriginWhenCrossOriginPolicy'},
            {},
            {'referrer_policy': 'ssscrapy-default'},
            OriginWhenCrossOriginPolicy,
            True
        ),
        (
            # same as above but with string value for settings policy
            {'REFERRER_POLICY': 'origin-when-cross-origin'},
            {},
            {'referrer_policy': 'ssscrapy-default'},
            OriginWhenCrossOriginPolicy,
            True
        ),
        (
            # request meta references a wrong policy but it is set,
            # so the Referrer-Policy header in response is not used,
            # and the settings' policy is applied
            {'REFERRER_POLICY': 'origin-when-cross-origin'},
            {'Referrer-Policy': 'unsafe-url'},
            {'referrer_policy': 'ssscrapy-default'},
            OriginWhenCrossOriginPolicy,
            True
        ),
        (
            # here, request meta does not set the policy
            # so response headers take precedence
            {'REFERRER_POLICY': 'origin-when-cross-origin'},
            {'Referrer-Policy': 'unsafe-url'},
            {},
            UnsafeUrlPolicy,
            False
        ),
        (
            # here, request meta does not set the policy,
            # but response headers also use an unknown policy,
            # so the settings' policy is used
            {'REFERRER_POLICY': 'origin-when-cross-origin'},
            {'Referrer-Policy': 'unknown'},
            {},
            OriginWhenCrossOriginPolicy,
            True
        )
    ]

    def test(self):

        origin = 'http://www.scrapy.org'
        target = 'http://www.example.com'

        for settings, response_headers, request_meta, policy_class, check_warning in self.params[3:]:
            spider = Spider('foo')
            mw = RefererMiddleware(Settings(settings))

            response = Response(origin, headers=response_headers)
            request = Request(target, meta=request_meta)

            with warnings.catch_warnings(record=True) as w:
                policy = mw.policy(response, request)
                self.assertIsInstance(policy, policy_class)

                if check_warning:
                    self.assertEqual(len(w), 1)
                    self.assertEqual(w[0].category, RuntimeWarning, w[0].message)


class EncodingResource(resource.Resource):
    out_encoding = 'cp1251'

    def render(self, request):
        body = to_unicode(request.content.read())
        request.setHeader(b'content-encoding', self.out_encoding)
        return body.encode(self.out_encoding)

