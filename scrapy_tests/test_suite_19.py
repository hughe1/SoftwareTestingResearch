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


class WrappedRequestTest(TestCase):

    def setUp(self):
        self.request = Request("http://www.example.com/page.html",
                               headers={"Content-Type": "text/html"})
        self.wrapped = WrappedRequest(self.request)

    def test_get_full_url(self):
        self.assertEqual(self.wrapped.get_full_url(), self.request.url)
        self.assertEqual(self.wrapped.full_url, self.request.url)

    def test_get_host(self):
        self.assertEqual(self.wrapped.get_host(), urlparse(self.request.url).netloc)
        self.assertEqual(self.wrapped.host, urlparse(self.request.url).netloc)

    def test_get_type(self):
        self.assertEqual(self.wrapped.get_type(), urlparse(self.request.url).scheme)
        self.assertEqual(self.wrapped.type, urlparse(self.request.url).scheme)

    def test_is_unverifiable(self):
        self.assertFalse(self.wrapped.is_unverifiable())
        self.assertFalse(self.wrapped.unverifiable)

    def test_is_unverifiable2(self):
        self.request.meta['is_unverifiable'] = True
        self.assertTrue(self.wrapped.is_unverifiable())
        self.assertTrue(self.wrapped.unverifiable)

    def test_get_origin_req_host(self):
        self.assertEqual(self.wrapped.get_origin_req_host(), 'www.example.com')
        self.assertEqual(self.wrapped.origin_req_host, 'www.example.com')

    def test_has_header(self):
        self.assertTrue(self.wrapped.has_header('content-type'))
        self.assertFalse(self.wrapped.has_header('xxxxx'))

    def test_get_header(self):
        self.assertEqual(self.wrapped.get_header('content-type'), 'text/html')
        self.assertEqual(self.wrapped.get_header('xxxxx', 'def'), 'def')

    def test_header_items(self):
        self.assertEqual(self.wrapped.header_items(),
                         [('Content-Type', ['text/html'])])

    def test_add_unredirected_header(self):
        self.wrapped.add_unredirected_header('hello', 'world')
        self.assertEqual(self.request.headers['hello'], b'world')


class DeprecatedImagesPipeline(ImagesPipeline):
    def file_key(self, url):
        return self.image_key(url)

    def image_key(self, url):
        image_guid = hashlib.sha1(to_bytes(url)).hexdigest()
        return 'empty/%s.jpg' % (image_guid)

    def thumb_key(self, url, thumb_id):
        thumb_guid = hashlib.sha1(to_bytes(url)).hexdigest()
        return 'thumbsup/%s/%s.jpg' % (thumb_id, thumb_guid)


class GenspiderCommandTest(CommandTest):

    def test_arguments(self):
        # only pass one argument. spider script shouldn't be created
        self.assertEqual(2, self.call('genspider', 'test_name'))
        assert not exists(join(self.proj_mod_path, 'spiders', 'test_name.py'))
        # pass two arguments <name> <domain>. spider script should be created
        self.assertEqual(0, self.call('genspider', 'test_name', 'test.com'))
        assert exists(join(self.proj_mod_path, 'spiders', 'test_name.py'))

    def test_template(self, tplname='crawl'):
        args = ['--template=%s' % tplname] if tplname else []
        spname = 'test_spider'
        p = self.proc('genspider', spname, 'test.com', *args)
        out = to_native_str(retry_on_eintr(p.stdout.read))
        self.assertIn("Created spider %r using template %r in module" % (spname, tplname), out)
        self.assertTrue(exists(join(self.proj_mod_path, 'spiders', 'test_spider.py')))
        p = self.proc('genspider', spname, 'test.com', *args)
        out = to_native_str(retry_on_eintr(p.stdout.read))
        self.assertIn("Spider %r already exists in module" % spname, out)

    def test_template_basic(self):
        self.test_template('basic')

    def test_template_csvfeed(self):
        self.test_template('csvfeed')

    def test_template_xmlfeed(self):
        self.test_template('xmlfeed')

    def test_list(self):
        self.assertEqual(0, self.call('genspider', '--list'))

    def test_dump(self):
        self.assertEqual(0, self.call('genspider', '--dump=basic'))
        self.assertEqual(0, self.call('genspider', '-d', 'basic'))

    def test_same_name_as_project(self):
        self.assertEqual(2, self.call('genspider', self.project_name))
        assert not exists(join(self.proj_mod_path, 'spiders', '%s.py' % self.project_name))


class TestPolicyHeaderPredecence004(MixinNoReferrerWhenDowngrade, TestRefererMiddleware):
    """
    The empty string means "no-referrer-when-downgrade"
    """
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.OriginWhenCrossOriginPolicy'}
    resp_headers = {'Referrer-Policy': ''}


class DeprecatedHttpTestCase(HttpTestCase):
    """HTTP 1.0 test case"""
    download_handler_cls = HttpDownloadHandler


class BrokenLinksMediaDownloadSpider(MediaDownloadSpider):
    name = 'brokenmedia'

    def _process_url(self, url):
        return url + '.foo'


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


class TestOffsiteMiddleware3(TestOffsiteMiddleware2):

    def _get_spider(self):
        return Spider('foo')


class MarshalLifoDiskQueueTest(t.LifoDiskQueueTest):

    def queue(self):
        return MarshalLifoDiskQueue(self.qpath)

    def test_serialize(self):
        q = self.queue()
        q.push('a')
        q.push(123)
        q.push({'a': 'dict'})
        self.assertEqual(q.pop(), {'a': 'dict'})
        self.assertEqual(q.pop(), 123)
        self.assertEqual(q.pop(), 'a')

    test_nonserializable_object = nonserializable_object_test


class BadSpider(scrapy.Spider):
    name = "bad"
    def start_requests(self):
        raise Exception("oops!")
        """, name="badspider.py")
        print(log)
        self.assertIn("start_requests", log)
        self.assertIn("badspider.py", log)


class ParseUrlTestCase(unittest.TestCase):
    """Test URL parsing facility and defaults values."""

    def _parse(self, url):
        f = client.ScrapyHTTPClientFactory(Request(url))
        return (f.scheme, f.netloc, f.host, f.port, f.path)

    def testParse(self):
        lip = '127.0.0.1'
        tests = (
    ("http://127.0.0.1?c=v&c2=v2#fragment",     ('http', lip, lip, 80, '/?c=v&c2=v2')),
    ("http://127.0.0.1/?c=v&c2=v2#fragment",    ('http', lip, lip, 80, '/?c=v&c2=v2')),
    ("http://127.0.0.1/foo?c=v&c2=v2#frag",     ('http', lip, lip, 80, '/foo?c=v&c2=v2')),
    ("http://127.0.0.1:100?c=v&c2=v2#fragment", ('http', lip+':100', lip, 100, '/?c=v&c2=v2')),
    ("http://127.0.0.1:100/?c=v&c2=v2#frag",    ('http', lip+':100', lip, 100, '/?c=v&c2=v2')),
    ("http://127.0.0.1:100/foo?c=v&c2=v2#frag", ('http', lip+':100', lip, 100, '/foo?c=v&c2=v2')),

    ("http://127.0.0.1",              ('http', lip, lip, 80, '/')),
    ("http://127.0.0.1/",             ('http', lip, lip, 80, '/')),
    ("http://127.0.0.1/foo",          ('http', lip, lip, 80, '/foo')),
    ("http://127.0.0.1?param=value",  ('http', lip, lip, 80, '/?param=value')),
    ("http://127.0.0.1/?param=value", ('http', lip, lip, 80, '/?param=value')),
    ("http://127.0.0.1:12345/foo",    ('http', lip+':12345', lip, 12345, '/foo')),
    ("http://spam:12345/foo",         ('http', 'spam:12345', 'spam', 12345, '/foo')),
    ("http://spam.test.org/foo",      ('http', 'spam.test.org', 'spam.test.org', 80, '/foo')),

    ("https://127.0.0.1/foo",         ('https', lip, lip, 443, '/foo')),
    ("https://127.0.0.1/?param=value", ('https', lip, lip, 443, '/?param=value')),
    ("https://127.0.0.1:12345/",      ('https', lip+':12345', lip, 12345, '/')),

    ("http://scrapytest.org/foo ",    ('http', 'scrapytest.org', 'scrapytest.org', 80, '/foo')),
    ("http://egg:7890 ",              ('http', 'egg:7890', 'egg', 7890, '/')),
    )

        for url, test in tests:
            test = tuple(
                to_bytes(x) if not isinstance(x, int) else x for x in test)
            self.assertEqual(client._parse(url), test, url)

    def test_externalUnicodeInterference(self):
        """
        L{client._parse} should return C{str} for the scheme, host, and path
        elements of its return tuple, even when passed an URL which has
        previously been passed to L{urlparse} as a C{unicode} string.
        """
        if not six.PY2:
            raise unittest.SkipTest(
                "Applies only to Py2, as urls can be ONLY unicode on Py3")
        badInput = u'http://example.com/path'
        goodInput = badInput.encode('ascii')
        self._parse(badInput)  # cache badInput in urlparse_cached
        scheme, netloc, host, port, path = self._parse(goodInput)
        self.assertTrue(isinstance(scheme, str))
        self.assertTrue(isinstance(netloc, str))
        self.assertTrue(isinstance(host, str))
        self.assertTrue(isinstance(path, str))
        self.assertTrue(isinstance(port, int))



class BaseCrawlerTest(unittest.TestCase):

    def assertOptionIsDefault(self, settings, key):
        self.assertIsInstance(settings, Settings)
        self.assertEqual(settings[key], getattr(default_settings, key))


class TestItem(Item):
    name = Field()
    url = Field()


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

class HttpCompressionTest(TestCase):

    def setUp(self):
        self.spider = Spider('foo')
        self.mw = HttpCompressionMiddleware()

    def _getresponse(self, coding):
        if coding not in FORMAT:
            raise ValueError()

        samplefile, contentencoding = FORMAT[coding]

        with open(join(SAMPLEDIR, samplefile), 'rb') as sample:
            body = sample.read()

        headers = {
                'Server': 'Yaws/1.49 Yet Another Web Server',
                'Date': 'Sun, 08 Mar 2009 00:41:03 GMT',
                'Content-Length': len(body),
                'Content-Type': 'text/html',
                'Content-Encoding': contentencoding,
                }

        response = Response('http://scrapytest.org/', body=body, headers=headers)
        response.request = Request('http://scrapytest.org', headers={'Accept-Encoding': 'gzip,deflate'})
        return response

    def test_process_request(self):
        request = Request('http://scrapytest.org')
        assert 'Accept-Encoding' not in request.headers
        self.mw.process_request(request, self.spider)
        self.assertEqual(request.headers.get('Accept-Encoding'),
                         b','.join(ACCEPTED_ENCODINGS))

    def test_process_response_gzip(self):
        response = self._getresponse('gzip')
        request = response.request

        self.assertEqual(response.headers['Content-Encoding'], b'gzip')
        newresponse = self.mw.process_response(request, response, self.spider)
        assert newresponse is not response
        assert newresponse.body.startswith(b'<!DOCTYPE')
        assert 'Content-Encoding' not in newresponse.headers

    def test_process_response_br(self):
        try:
            import brotli
        except ImportError:
            raise SkipTest("no brotli")
        response = self._getresponse('br')
        request = response.request
        self.assertEqual(response.headers['Content-Encoding'], b'br')
        newresponse = self.mw.process_response(request, response, self.spider)
        assert newresponse is not response
        assert newresponse.body.startswith(b"<!DOCTYPE")
        assert 'Content-Encoding' not in newresponse.headers

    def test_process_response_rawdeflate(self):
        response = self._getresponse('rawdeflate')
        request = response.request

        self.assertEqual(response.headers['Content-Encoding'], b'deflate')
        newresponse = self.mw.process_response(request, response, self.spider)
        assert newresponse is not response
        assert newresponse.body.startswith(b'<!DOCTYPE')
        assert 'Content-Encoding' not in newresponse.headers

    def test_process_response_zlibdelate(self):
        response = self._getresponse('zlibdeflate')
        request = response.request

        self.assertEqual(response.headers['Content-Encoding'], b'deflate')
        newresponse = self.mw.process_response(request, response, self.spider)
        assert newresponse is not response
        assert newresponse.body.startswith(b'<!DOCTYPE')
        assert 'Content-Encoding' not in newresponse.headers

    def test_process_response_plain(self):
        response = Response('http://scrapytest.org', body=b'<!DOCTYPE...')
        request = Request('http://scrapytest.org')

        assert not response.headers.get('Content-Encoding')
        newresponse = self.mw.process_response(request, response, self.spider)
        assert newresponse is response
        assert newresponse.body.startswith(b'<!DOCTYPE')

    def test_multipleencodings(self):
        response = self._getresponse('gzip')
        response.headers['Content-Encoding'] = ['uuencode', 'gzip']
        request = response.request
        newresponse = self.mw.process_response(request, response, self.spider)
        assert newresponse is not response
        self.assertEqual(newresponse.headers.getlist('Content-Encoding'), [b'uuencode'])

    def test_process_response_encoding_inside_body(self):
        headers = {
            'Content-Type': 'text/html',
            'Content-Encoding': 'gzip',
        }
        f = BytesIO()
        plainbody = b"""<html><head><title>Some page</title><meta http-equiv="Content-Type" content="text/html; charset=gb2312">"""
        zf = GzipFile(fileobj=f, mode='wb')
        zf.write(plainbody)
        zf.close()
        response = Response("http;//www.example.com/", headers=headers, body=f.getvalue())
        request = Request("http://www.example.com/")

        newresponse = self.mw.process_response(request, response, self.spider)
        assert isinstance(newresponse, HtmlResponse)
        self.assertEqual(newresponse.body, plainbody)
        self.assertEqual(newresponse.encoding, resolve_encoding('gb2312'))

    def test_process_response_force_recalculate_encoding(self):
        headers = {
            'Content-Type': 'text/html',
            'Content-Encoding': 'gzip',
        }
        f = BytesIO()
        plainbody = b"""<html><head><title>Some page</title><meta http-equiv="Content-Type" content="text/html; charset=gb2312">"""
        zf = GzipFile(fileobj=f, mode='wb')
        zf.write(plainbody)
        zf.close()
        response = HtmlResponse("http;//www.example.com/page.html", headers=headers, body=f.getvalue())
        request = Request("http://www.example.com/")

        newresponse = self.mw.process_response(request, response, self.spider)
        assert isinstance(newresponse, HtmlResponse)
        self.assertEqual(newresponse.body, plainbody)
        self.assertEqual(newresponse.encoding, resolve_encoding('gb2312'))

    def test_process_response_no_content_type_header(self):
        headers = {
            'Content-Encoding': 'identity',
        }
        plainbody = b"""<html><head><title>Some page</title><meta http-equiv="Content-Type" content="text/html; charset=gb2312">"""
        respcls = responsetypes.from_args(url="http://www.example.com/index", headers=headers, body=plainbody)
        response = respcls("http://www.example.com/index", headers=headers, body=plainbody)
        request = Request("http://www.example.com/index")

        newresponse = self.mw.process_response(request, response, self.spider)
        assert isinstance(newresponse, respcls)
        self.assertEqual(newresponse.body, plainbody)
        self.assertEqual(newresponse.encoding, resolve_encoding('gb2312'))

    def test_process_response_gzipped_contenttype(self):
        response = self._getresponse('gzip')
        response.headers['Content-Type'] = 'application/gzip'
        request = response.request

        newresponse = self.mw.process_response(request, response, self.spider)
        self.assertIsNot(newresponse, response)
        self.assertTrue(newresponse.body.startswith(b'<!DOCTYPE'))
        self.assertNotIn('Content-Encoding', newresponse.headers)

    def test_process_response_gzip_app_octetstream_contenttype(self):
        response = self._getresponse('gzip')
        response.headers['Content-Type'] = 'application/octet-stream'
        request = response.request

        newresponse = self.mw.process_response(request, response, self.spider)
        self.assertIsNot(newresponse, response)
        self.assertTrue(newresponse.body.startswith(b'<!DOCTYPE'))
        self.assertNotIn('Content-Encoding', newresponse.headers)

    def test_process_response_gzip_binary_octetstream_contenttype(self):
        response = self._getresponse('x-gzip')
        response.headers['Content-Type'] = 'binary/octet-stream'
        request = response.request

        newresponse = self.mw.process_response(request, response, self.spider)
        self.assertIsNot(newresponse, response)
        self.assertTrue(newresponse.body.startswith(b'<!DOCTYPE'))
        self.assertNotIn('Content-Encoding', newresponse.headers)

    def test_process_response_gzipped_gzip_file(self):
        """Test that a gzip Content-Encoded .gz file is gunzipped
        only once by the middleware, leaving gunzipping of the file
        to upper layers.
        """
        headers = {
            'Content-Type': 'application/gzip',
            'Content-Encoding': 'gzip',
        }
        # build a gzipped file (here, a sitemap)
        f = BytesIO()
        plainbody = b"""<?xml version="1.0" encoding="UTF-8"?>
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
</urlset>"""
        gz_file = GzipFile(fileobj=f, mode='wb')
        gz_file.write(plainbody)
        gz_file.close()

        # build a gzipped response body containing this gzipped file
        r = BytesIO()
        gz_resp = GzipFile(fileobj=r, mode='wb')
        gz_resp.write(f.getvalue())
        gz_resp.close()

        response = Response("http;//www.example.com/", headers=headers, body=r.getvalue())
        request = Request("http://www.example.com/")

        newresponse = self.mw.process_response(request, response, self.spider)
        self.assertEqual(gunzip(newresponse.body), plainbody)

    def test_process_response_head_request_no_decode_required(self):
        response = self._getresponse('gzip')
        response.headers['Content-Type'] = 'application/gzip'
        request = response.request
        request.method = 'HEAD'
        response = response.replace(body = None)
        newresponse = self.mw.process_response(request, response, self.spider)
        self.assertIs(newresponse, response)
        self.assertEqual(response.body, b'')

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


class StreamLoggerTest(unittest.TestCase):

    def setUp(self):
        self.stdout = sys.stdout
        logger = logging.getLogger('test')
        logger.setLevel(logging.WARNING)
        sys.stdout = StreamLogger(logger, logging.ERROR)

    def tearDown(self):
        sys.stdout = self.stdout

    def test_redirect(self):
        with LogCapture() as l:
            print('test log msg')
        l.check(('test', 'ERROR', 'test log msg'))

class RedirectedMediaDownloadSpider(MediaDownloadSpider):
    name = 'redirectedmedia'

    def _process_url(self, url):
        return add_or_replace_parameter(
                    'http://localhost:8998/redirect-to',
                    'goto', url)


class ChunkSize1MarshalFifoDiskQueueTest(MarshalFifoDiskQueueTest):
    chunksize = 1

class BlockingFeedStorageTest(unittest.TestCase):

    def get_test_spider(self, settings=None):
        class TestSpider(scrapy.Spider):
            name = 'test_spider'
        crawler = get_crawler(settings_dict=settings)
        spider = TestSpider.from_crawler(crawler)
        return spider

    def test_default_temp_dir(self):
        b = BlockingFeedStorage()

        tmp = b.open(self.get_test_spider())
        tmp_path = os.path.dirname(tmp.name)
        self.assertEqual(tmp_path, tempfile.gettempdir())

    def test_temp_file(self):
        b = BlockingFeedStorage()

        tests_path = os.path.dirname(os.path.abspath(__file__))
        spider = self.get_test_spider({'FEED_TEMPDIR': tests_path})
        tmp = b.open(spider)
        tmp_path = os.path.dirname(tmp.name)
        self.assertEqual(tmp_path, tests_path)

    def test_invalid_folder(self):
        b = BlockingFeedStorage()

        tests_path = os.path.dirname(os.path.abspath(__file__))
        invalid_path = os.path.join(tests_path, 'invalid_path')
        spider = self.get_test_spider({'FEED_TEMPDIR': invalid_path})

        self.assertRaises(OSError, b.open, spider=spider)


class CustomItem(Item):

    name = Field()

    def __str__(self):
        return "name: %s" % self['name']


class MixinStrictOriginWhenCrossOrigin(object):
    scenarii = [
        # Same origin (protocol, host, port): send referrer
        ('https://example.com/page.html',       'https://example.com/not-page.html',        b'https://example.com/page.html'),
        ('http://example.com/page.html',        'http://example.com/not-page.html',         b'http://example.com/page.html'),
        ('https://example.com:443/page.html',   'https://example.com/not-page.html',        b'https://example.com/page.html'),
        ('http://example.com:80/page.html',     'http://example.com/not-page.html',         b'http://example.com/page.html'),
        ('http://example.com/page.html',        'http://example.com:80/not-page.html',      b'http://example.com/page.html'),
        ('http://example.com:8888/page.html',   'http://example.com:8888/not-page.html',    b'http://example.com:8888/page.html'),

        # Different host: send origin as referrer
        ('https://example2.com/page.html',  'https://scrapy.org/otherpage.html',        b'https://example2.com/'),
        ('https://example2.com/page.html',  'https://not.example2.com/otherpage.html',  b'https://example2.com/'),
        ('http://example2.com/page.html',   'http://not.example2.com/otherpage.html',   b'http://example2.com/'),
        # exact match required
        ('http://example2.com/page.html',   'http://www.example2.com/otherpage.html',   b'http://example2.com/'),

        # Different port: send origin as referrer
        ('https://example3.com:444/page.html',  'https://example3.com/not-page.html',   b'https://example3.com:444/'),
        ('http://example3.com:81/page.html',    'http://example3.com/not-page.html',    b'http://example3.com:81/'),

        # downgrade
        ('https://example4.com/page.html',  'http://example4.com/not-page.html',    None),
        ('https://example4.com/page.html',  'http://not.example4.com/',             None),

        # non-TLS to non-TLS
        ('ftp://example4.com/urls.zip',     'http://example4.com/not-page.html',    b'ftp://example4.com/'),

        # upgrade
        ('http://example4.com/page.html',  'https://example4.com/not-page.html',    b'http://example4.com/'),
        ('http://example4.com/page.html',  'https://not.example4.com/',             b'http://example4.com/'),

        # Different protocols: send origin as referrer
        ('ftps://example4.com/urls.zip',    'https://example4.com/not-page.html',   b'ftps://example4.com/'),
        ('ftps://example4.com/urls.zip',    'https://example4.com/not-page.html',   b'ftps://example4.com/'),

        # test for user/password stripping
        ('https://user:password@example5.com/page.html', 'https://example5.com/not-page.html',  b'https://example5.com/page.html'),

        # TLS to non-TLS downgrade: send nothing
        ('https://user:password@example5.com/page.html', 'http://example5.com/not-page.html',   None),
    ]


class ChunkedTest(unittest.TestCase):

    def test_decode_chunked_transfer(self):
        """Example taken from: http://en.wikipedia.org/wiki/Chunked_transfer_encoding"""
        chunked_body = "25\r\n" + "This is the data in the first chunk\r\n\r\n"
        chunked_body += "1C\r\n" + "and this is the second one\r\n\r\n"
        chunked_body += "3\r\n" + "con\r\n"
        chunked_body += "8\r\n" + "sequence\r\n"
        chunked_body += "0\r\n\r\n"
        body = decode_chunked_transfer(chunked_body)
        self.assertEqual(body, \
            "This is the data in the first chunk\r\n" +
            "and this is the second one\r\n" +
            "consequence")



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


class DeprecatedFilesPipeline(FilesPipeline):
    def file_key(self, url):
        media_guid = hashlib.sha1(to_bytes(url)).hexdigest()
        media_ext = os.path.splitext(url)[1]
        return 'empty/%s%s' % (media_guid, media_ext)


class StreamLoggerTest(unittest.TestCase):

    def setUp(self):
        self.stdout = sys.stdout
        logger = logging.getLogger('test')
        logger.setLevel(logging.WARNING)
        sys.stdout = StreamLogger(logger, logging.ERROR)

    def tearDown(self):
        sys.stdout = self.stdout

    def test_redirect(self):
        with LogCapture() as l:
            print('test log msg')
        l.check(('test', 'ERROR', 'test log msg'))

class TestSettingsCustomPolicy(TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'tests.test_spidermiddleware_referer.CustomPythonOrgPolicy'}
    scenarii = [
        ('https://example.com/',    'https://scrapy.org/',  b'https://python.org/'),
        ('http://example.com/',     'http://scrapy.org/',   b'http://python.org/'),
        ('http://example.com/',     'https://scrapy.org/',  b'https://python.org/'),
        ('https://example.com/',    'http://scrapy.org/',   b'http://python.org/'),
        ('file:///home/path/to/somefile.html',  'https://scrapy.org/', b'https://python.org/'),
        ('file:///home/path/to/somefile.html',  'http://scrapy.org/',  b'http://python.org/'),

    ]

# --- Tests using Request meta dict to set policy
class PprintItemExporterTest(BaseItemExporterTest):

    def _get_exporter(self, **kwargs):
        return PprintItemExporter(self.output, **kwargs)

    def _check_output(self):
        self._assert_expected_item(eval(self.output.getvalue()))


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

