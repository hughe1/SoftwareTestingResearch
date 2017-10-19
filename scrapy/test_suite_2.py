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


class BrokenChunkedResource(resource.Resource):

    def render(self, request):
        def response():
            request.write(b"chunked ")
            request.write(b"content\n")
            # Disable terminating chunk on finish.
            request.chunked = False
            closeConnection(request)
        reactor.callLater(0, response)
        return server.NOT_DONE_YET


class ContentLengthHeaderResource(resource.Resource):
    """
    A testing resource which renders itself as the value of the Content-Length
    header from the request.
    """
    def render(self, request):
        return request.requestHeaders.getRawHeaders(b"content-length")[0]


class DeprecatedHttpTestCase(HttpTestCase):
    """HTTP 1.0 test case"""
    download_handler_cls = HttpDownloadHandler


class DeprecatedHttpProxyTestCase(unittest.TestCase):
    """Old deprecated reference to http10 downloader handler"""
    download_handler_cls = HttpDownloadHandler


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

class DeprecatedXpathSelectorTest(unittest.TestCase):

    text = '<div><img src="a.jpg"><p>Hello</div>'

    def test_warnings_xpathselector(self):
        cls = XPathSelector
        with warnings.catch_warnings(record=True) as w:
            class UserClass(cls):
                pass

            # subclassing must issue a warning
            self.assertEqual(len(w), 1, str(cls))
            self.assertIn('scrapy.Selector', str(w[0].message))

            # subclass instance doesn't issue a warning
            usel = UserClass(text=self.text)
            self.assertEqual(len(w), 1)

            # class instance must issue a warning
            sel = cls(text=self.text)
            self.assertEqual(len(w), 2, str((cls, [x.message for x in w])))
            self.assertIn('scrapy.Selector', str(w[1].message))

            # subclass and instance checks
            self.assertTrue(issubclass(cls, Selector))
            self.assertTrue(isinstance(sel, Selector))
            self.assertTrue(isinstance(usel, Selector))

    def test_warnings_xmlxpathselector(self):
        cls = XmlXPathSelector
        with warnings.catch_warnings(record=True) as w:
            class UserClass(cls):
                pass

            # subclassing must issue a warning
            self.assertEqual(len(w), 1, str(cls))
            self.assertIn('scrapy.Selector', str(w[0].message))

            # subclass instance doesn't issue a warning
            usel = UserClass(text=self.text)
            self.assertEqual(len(w), 1)

            # class instance must issue a warning
            sel = cls(text=self.text)
            self.assertEqual(len(w), 2, str((cls, [x.message for x in w])))
            self.assertIn('scrapy.Selector', str(w[1].message))

            # subclass and instance checks
            self.assertTrue(issubclass(cls, Selector))
            self.assertTrue(issubclass(cls, XPathSelector))
            self.assertTrue(isinstance(sel, Selector))
            self.assertTrue(isinstance(usel, Selector))
            self.assertTrue(isinstance(sel, XPathSelector))
            self.assertTrue(isinstance(usel, XPathSelector))

    def test_warnings_htmlxpathselector(self):
        cls = HtmlXPathSelector
        with warnings.catch_warnings(record=True) as w:
            class UserClass(cls):
                pass

            # subclassing must issue a warning
            self.assertEqual(len(w), 1, str(cls))
            self.assertIn('scrapy.Selector', str(w[0].message))

            # subclass instance doesn't issue a warning
            usel = UserClass(text=self.text)
            self.assertEqual(len(w), 1)

            # class instance must issue a warning
            sel = cls(text=self.text)
            self.assertEqual(len(w), 2, str((cls, [x.message for x in w])))
            self.assertIn('scrapy.Selector', str(w[1].message))

            # subclass and instance checks
            self.assertTrue(issubclass(cls, Selector))
            self.assertTrue(issubclass(cls, XPathSelector))
            self.assertTrue(isinstance(sel, Selector))
            self.assertTrue(isinstance(usel, Selector))
            self.assertTrue(isinstance(sel, XPathSelector))
            self.assertTrue(isinstance(usel, XPathSelector))

class MixinUnsafeUrl(object):
    scenarii = [
        # TLS to TLS: send referrer
        ('https://example.com/sekrit.html',     'http://not.example.com/',      b'https://example.com/sekrit.html'),
        ('https://example1.com/page.html',      'https://not.example1.com/',    b'https://example1.com/page.html'),
        ('https://example1.com/page.html',      'https://scrapy.org/',          b'https://example1.com/page.html'),
        ('https://example1.com:443/page.html',  'https://scrapy.org/',          b'https://example1.com/page.html'),
        ('https://example1.com:444/page.html',  'https://scrapy.org/',          b'https://example1.com:444/page.html'),
        ('ftps://example1.com/urls.zip',        'https://scrapy.org/',          b'ftps://example1.com/urls.zip'),

        # TLS to non-TLS: send referrer (yes, it's unsafe)
        ('https://example2.com/page.html',  'http://not.example2.com/', b'https://example2.com/page.html'),
        ('https://example2.com/page.html',  'http://scrapy.org/',       b'https://example2.com/page.html'),
        ('ftps://example2.com/urls.zip',    'http://scrapy.org/',       b'ftps://example2.com/urls.zip'),

        # non-TLS to TLS or non-TLS: send referrer (yes, it's unsafe)
        ('http://example3.com/page.html',       'https://not.example3.com/',    b'http://example3.com/page.html'),
        ('http://example3.com/page.html',       'https://scrapy.org/',          b'http://example3.com/page.html'),
        ('http://example3.com:8080/page.html',  'https://scrapy.org/',          b'http://example3.com:8080/page.html'),
        ('http://example3.com:80/page.html',    'http://not.example3.com/',     b'http://example3.com/page.html'),
        ('http://example3.com/page.html',       'http://scrapy.org/',           b'http://example3.com/page.html'),
        ('http://example3.com:443/page.html',   'http://scrapy.org/',           b'http://example3.com:443/page.html'),
        ('ftp://example3.com/urls.zip',         'http://scrapy.org/',           b'ftp://example3.com/urls.zip'),
        ('ftp://example3.com/urls.zip',         'https://scrapy.org/',          b'ftp://example3.com/urls.zip'),

        # test for user/password stripping
        ('http://user:password@example4.com/page.html',     'https://not.example4.com/',    b'http://example4.com/page.html'),
        ('https://user:password@example4.com/page.html',    'http://scrapy.org/',           b'https://example4.com/page.html'),
    ]


class FilesystemStorageGzipTest(FilesystemStorageTest):

    def _get_settings(self, **new_settings):
        new_settings.setdefault('HTTPCACHE_GZIP', True)
        return super(FilesystemStorageTest, self)._get_settings(**new_settings)

class BaseCrawlerTest(unittest.TestCase):

    def assertOptionIsDefault(self, settings, key):
        self.assertIsInstance(settings, Settings)
        self.assertEqual(settings[key], getattr(default_settings, key))


class LogFormatterSubclass(LogFormatter):
    def crawled(self, request, response, spider):
        kwargs = super(LogFormatterSubclass, self).crawled(
        request, response, spider)
        CRAWLEDMSG = (
            u"Crawled (%(status)s) %(request)s (referer: "
            u"%(referer)s)%(flags)s"
        )
        return {
            'level': kwargs['level'],
            'msg': CRAWLEDMSG,
            'args': kwargs['args']
        }


class MySpider2(MyBaseSpider):
    name = 'myspider2'

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


class SomeBaseClass(object):
    pass


class TestRequestMetaStrictOriginWhenCrossOrigin(MixinStrictOriginWhenCrossOrigin, TestRefererMiddleware):
    req_meta = {'referrer_policy': POLICY_STRICT_ORIGIN_WHEN_CROSS_ORIGIN}


class M2(object):

    def open_spider(self, spider):
        pass

    def close_spider(self, spider):
        pass

    pass

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



class UtilsSpidersTestCase(unittest.TestCase):

    def test_iterate_spider_output(self):
        i = BaseItem()
        r = Request('http://scrapytest.org')
        o = object()

        self.assertEqual(list(iterate_spider_output(i)), [i])
        self.assertEqual(list(iterate_spider_output(r)), [r])
        self.assertEqual(list(iterate_spider_output(o)), [o])
        self.assertEqual(list(iterate_spider_output([r, i, o])), [r, i, o])

    def test_iter_spider_classes(self):
        import tests.test_utils_spider
        it = iter_spider_classes(tests.test_utils_spider)
        self.assertEqual(set(it), {MySpider1, MySpider2})

if __name__ == "__main__":
    unittest.main()


class SomeBaseClass(object):
    pass


class MixinSameOrigin(object):
    scenarii = [
        # Same origin (protocol, host, port): send referrer
        ('https://example.com/page.html',       'https://example.com/not-page.html',        b'https://example.com/page.html'),
        ('http://example.com/page.html',        'http://example.com/not-page.html',         b'http://example.com/page.html'),
        ('https://example.com:443/page.html',   'https://example.com/not-page.html',        b'https://example.com/page.html'),
        ('http://example.com:80/page.html',     'http://example.com/not-page.html',         b'http://example.com/page.html'),
        ('http://example.com/page.html',        'http://example.com:80/not-page.html',      b'http://example.com/page.html'),
        ('http://example.com:8888/page.html',   'http://example.com:8888/not-page.html',    b'http://example.com:8888/page.html'),

        # Different host: do NOT send referrer
        ('https://example.com/page.html',       'https://not.example.com/otherpage.html',   None),
        ('http://example.com/page.html',        'http://not.example.com/otherpage.html',    None),
        ('http://example.com/page.html',        'http://www.example.com/otherpage.html',    None),

        # Different port: do NOT send referrer
        ('https://example.com:444/page.html',   'https://example.com/not-page.html',    None),
        ('http://example.com:81/page.html',     'http://example.com/not-page.html',     None),
        ('http://example.com/page.html',        'http://example.com:81/not-page.html',  None),

        # Different protocols: do NOT send refferer
        ('https://example.com/page.html',   'http://example.com/not-page.html',     None),
        ('https://example.com/page.html',   'http://not.example.com/',              None),
        ('ftps://example.com/urls.zip',     'https://example.com/not-page.html',    None),
        ('ftp://example.com/urls.zip',      'http://example.com/not-page.html',     None),
        ('ftps://example.com/urls.zip',     'https://example.com/not-page.html',    None),

        # test for user/password stripping
        ('https://user:password@example.com/page.html', 'https://example.com/not-page.html',    b'https://example.com/page.html'),
        ('https://user:password@example.com/page.html', 'http://example.com/not-page.html',     None),
    ]


class BasicItemLoaderTest(unittest.TestCase):

    def test_load_item_using_default_loader(self):
        i = TestItem()
        i['summary'] = u'lala'
        il = ItemLoader(item=i)
        il.add_value('name', u'marta')
        item = il.load_item()
        assert item is i
        self.assertEqual(item['summary'], u'lala')
        self.assertEqual(item['name'], [u'marta'])

    def test_load_item_using_custom_loader(self):
        il = TestItemLoader()
        il.add_value('name', u'marta')
        item = il.load_item()
        self.assertEqual(item['name'], [u'Marta'])

    def test_load_item_ignore_none_field_values(self):
        def validate_sku(value):
            # Let's assume a SKU is only digits.
            if value.isdigit():
                return value

        class MyLoader(ItemLoader):
            name_out = Compose(lambda vs: vs[0])  # take first which allows empty values
            price_out = Compose(TakeFirst(), float)
            sku_out = Compose(TakeFirst(), validate_sku)

        valid_fragment = u'SKU: 1234'
        invalid_fragment = u'SKU: not available'
        sku_re = 'SKU: (.+)'

        il = MyLoader(item={})
        # Should not return "sku: None".
        il.add_value('sku', [invalid_fragment], re=sku_re)
        # Should not ignore empty values.
        il.add_value('name', u'')
        il.add_value('price', [u'0'])
        self.assertEqual(il.load_item(), {
            'name': u'',
            'price': 0.0,
        })

        il.replace_value('sku', [valid_fragment], re=sku_re)
        self.assertEqual(il.load_item()['sku'], u'1234')

    def test_self_referencing_loader(self):
        class MyLoader(ItemLoader):
            url_out = TakeFirst()

            def img_url_out(self, values):
                return (self.get_output_value('url') or '') + values[0]

        il = MyLoader(item={})
        il.add_value('url', 'http://example.com/')
        il.add_value('img_url', '1234.png')
        self.assertEqual(il.load_item(), {
            'url': 'http://example.com/',
            'img_url': 'http://example.com/1234.png',
        })

        il = MyLoader(item={})
        il.add_value('img_url', '1234.png')
        self.assertEqual(il.load_item(), {
            'img_url': '1234.png',
        })

    def test_add_value(self):
        il = TestItemLoader()
        il.add_value('name', u'marta')
        self.assertEqual(il.get_collected_values('name'), [u'Marta'])
        self.assertEqual(il.get_output_value('name'), [u'Marta'])
        il.add_value('name', u'pepe')
        self.assertEqual(il.get_collected_values('name'), [u'Marta', u'Pepe'])
        self.assertEqual(il.get_output_value('name'), [u'Marta', u'Pepe'])

        # test add object value
        il.add_value('summary', {'key': 1})
        self.assertEqual(il.get_collected_values('summary'), [{'key': 1}])

        il.add_value(None, u'Jim', lambda x: {'name': x})
        self.assertEqual(il.get_collected_values('name'), [u'Marta', u'Pepe', u'Jim'])

    def test_add_zero(self):
        il = NameItemLoader()
        il.add_value('name', 0)
        self.assertEqual(il.get_collected_values('name'), [0])

    def test_replace_value(self):
        il = TestItemLoader()
        il.replace_value('name', u'marta')
        self.assertEqual(il.get_collected_values('name'), [u'Marta'])
        self.assertEqual(il.get_output_value('name'), [u'Marta'])
        il.replace_value('name', u'pepe')
        self.assertEqual(il.get_collected_values('name'), [u'Pepe'])
        self.assertEqual(il.get_output_value('name'), [u'Pepe'])

        il.replace_value(None, u'Jim', lambda x: {'name': x})
        self.assertEqual(il.get_collected_values('name'), [u'Jim'])

    def test_get_value(self):
        il = NameItemLoader()
        self.assertEqual(u'FOO', il.get_value([u'foo', u'bar'], TakeFirst(), six.text_type.upper))
        self.assertEqual([u'foo', u'bar'], il.get_value([u'name:foo', u'name:bar'], re=u'name:(.*)$'))
        self.assertEqual(u'foo', il.get_value([u'name:foo', u'name:bar'], TakeFirst(), re=u'name:(.*)$'))

        il.add_value('name', [u'name:foo', u'name:bar'], TakeFirst(), re=u'name:(.*)$')
        self.assertEqual([u'foo'], il.get_collected_values('name'))
        il.replace_value('name', u'name:bar', re=u'name:(.*)$')
        self.assertEqual([u'bar'], il.get_collected_values('name'))

    def test_iter_on_input_processor_input(self):
        class NameFirstItemLoader(NameItemLoader):
            name_in = TakeFirst()

        il = NameFirstItemLoader()
        il.add_value('name', u'marta')
        self.assertEqual(il.get_collected_values('name'), [u'marta'])
        il = NameFirstItemLoader()
        il.add_value('name', [u'marta', u'jose'])
        self.assertEqual(il.get_collected_values('name'), [u'marta'])

        il = NameFirstItemLoader()
        il.replace_value('name', u'marta')
        self.assertEqual(il.get_collected_values('name'), [u'marta'])
        il = NameFirstItemLoader()
        il.replace_value('name', [u'marta', u'jose'])
        self.assertEqual(il.get_collected_values('name'), [u'marta'])

        il = NameFirstItemLoader()
        il.add_value('name', u'marta')
        il.add_value('name', [u'jose', u'pedro'])
        self.assertEqual(il.get_collected_values('name'), [u'marta', u'jose'])

    def test_map_compose_filter(self):
        def filter_world(x):
            return None if x == 'world' else x

        proc = MapCompose(filter_world, str.upper)
        self.assertEqual(proc(['hello', 'world', 'this', 'is', 'scrapy']),
                         ['HELLO', 'THIS', 'IS', 'SCRAPY'])

    def test_map_compose_filter_multil(self):
        class TestItemLoader(NameItemLoader):
            name_in = MapCompose(lambda v: v.title(), lambda v: v[:-1])

        il = TestItemLoader()
        il.add_value('name', u'marta')
        self.assertEqual(il.get_output_value('name'), [u'Mart'])
        item = il.load_item()
        self.assertEqual(item['name'], [u'Mart'])

    def test_default_input_processor(self):
        il = DefaultedItemLoader()
        il.add_value('name', u'marta')
        self.assertEqual(il.get_output_value('name'), [u'mart'])

    def test_inherited_default_input_processor(self):
        class InheritDefaultedItemLoader(DefaultedItemLoader):
            pass

        il = InheritDefaultedItemLoader()
        il.add_value('name', u'marta')
        self.assertEqual(il.get_output_value('name'), [u'mart'])

    def test_input_processor_inheritance(self):
        class ChildItemLoader(TestItemLoader):
            url_in = MapCompose(lambda v: v.lower())

        il = ChildItemLoader()
        il.add_value('url', u'HTTP://scrapy.ORG')
        self.assertEqual(il.get_output_value('url'), [u'http://scrapy.org'])
        il.add_value('name', u'marta')
        self.assertEqual(il.get_output_value('name'), [u'Marta'])

        class ChildChildItemLoader(ChildItemLoader):
            url_in = MapCompose(lambda v: v.upper())
            summary_in = MapCompose(lambda v: v)

        il = ChildChildItemLoader()
        il.add_value('url', u'http://scrapy.org')
        self.assertEqual(il.get_output_value('url'), [u'HTTP://SCRAPY.ORG'])
        il.add_value('name', u'marta')
        self.assertEqual(il.get_output_value('name'), [u'Marta'])

    def test_empty_map_compose(self):
        class IdentityDefaultedItemLoader(DefaultedItemLoader):
            name_in = MapCompose()

        il = IdentityDefaultedItemLoader()
        il.add_value('name', u'marta')
        self.assertEqual(il.get_output_value('name'), [u'marta'])

    def test_identity_input_processor(self):
        class IdentityDefaultedItemLoader(DefaultedItemLoader):
            name_in = Identity()

        il = IdentityDefaultedItemLoader()
        il.add_value('name', u'marta')
        self.assertEqual(il.get_output_value('name'), [u'marta'])

    def test_extend_custom_input_processors(self):
        class ChildItemLoader(TestItemLoader):
            name_in = MapCompose(TestItemLoader.name_in, six.text_type.swapcase)

        il = ChildItemLoader()
        il.add_value('name', u'marta')
        self.assertEqual(il.get_output_value('name'), [u'mARTA'])

    def test_extend_default_input_processors(self):
        class ChildDefaultedItemLoader(DefaultedItemLoader):
            name_in = MapCompose(DefaultedItemLoader.default_input_processor, six.text_type.swapcase)

        il = ChildDefaultedItemLoader()
        il.add_value('name', u'marta')
        self.assertEqual(il.get_output_value('name'), [u'MART'])

    def test_output_processor_using_function(self):
        il = TestItemLoader()
        il.add_value('name', [u'mar', u'ta'])
        self.assertEqual(il.get_output_value('name'), [u'Mar', u'Ta'])

        class TakeFirstItemLoader(TestItemLoader):
            name_out = u" ".join

        il = TakeFirstItemLoader()
        il.add_value('name', [u'mar', u'ta'])
        self.assertEqual(il.get_output_value('name'), u'Mar Ta')

    def test_output_processor_error(self):
        class TestItemLoader(ItemLoader):
            default_item_class = TestItem
            name_out = MapCompose(float)

        il = TestItemLoader()
        il.add_value('name', [u'$10'])
        try:
            float(u'$10')
        except Exception as e:
            expected_exc_str = str(e)

        exc = None
        try:
            il.load_item()
        except Exception as e:
            exc = e
        assert isinstance(exc, ValueError)
        s = str(exc)
        assert 'name' in s, s
        assert '$10' in s, s
        assert 'ValueError' in s, s
        assert expected_exc_str in s, s

    def test_output_processor_using_classes(self):
        il = TestItemLoader()
        il.add_value('name', [u'mar', u'ta'])
        self.assertEqual(il.get_output_value('name'), [u'Mar', u'Ta'])

        class TakeFirstItemLoader(TestItemLoader):
            name_out = Join()

        il = TakeFirstItemLoader()
        il.add_value('name', [u'mar', u'ta'])
        self.assertEqual(il.get_output_value('name'), u'Mar Ta')

        class TakeFirstItemLoader(TestItemLoader):
            name_out = Join("<br>")

        il = TakeFirstItemLoader()
        il.add_value('name', [u'mar', u'ta'])
        self.assertEqual(il.get_output_value('name'), u'Mar<br>Ta')

    def test_default_output_processor(self):
        il = TestItemLoader()
        il.add_value('name', [u'mar', u'ta'])
        self.assertEqual(il.get_output_value('name'), [u'Mar', u'Ta'])

        class LalaItemLoader(TestItemLoader):
            default_output_processor = Identity()

        il = LalaItemLoader()
        il.add_value('name', [u'mar', u'ta'])
        self.assertEqual(il.get_output_value('name'), [u'Mar', u'Ta'])

    def test_loader_context_on_declaration(self):
        class ChildItemLoader(TestItemLoader):
            url_in = MapCompose(processor_with_args, key=u'val')

        il = ChildItemLoader()
        il.add_value('url', u'text')
        self.assertEqual(il.get_output_value('url'), ['val'])
        il.replace_value('url', u'text2')
        self.assertEqual(il.get_output_value('url'), ['val'])

    def test_loader_context_on_instantiation(self):
        class ChildItemLoader(TestItemLoader):
            url_in = MapCompose(processor_with_args)

        il = ChildItemLoader(key=u'val')
        il.add_value('url', u'text')
        self.assertEqual(il.get_output_value('url'), ['val'])
        il.replace_value('url', u'text2')
        self.assertEqual(il.get_output_value('url'), ['val'])

    def test_loader_context_on_assign(self):
        class ChildItemLoader(TestItemLoader):
            url_in = MapCompose(processor_with_args)

        il = ChildItemLoader()
        il.context['key'] = u'val'
        il.add_value('url', u'text')
        self.assertEqual(il.get_output_value('url'), ['val'])
        il.replace_value('url', u'text2')
        self.assertEqual(il.get_output_value('url'), ['val'])

    def test_item_passed_to_input_processor_functions(self):
        def processor(value, loader_context):
            return loader_context['item']['name']

        class ChildItemLoader(TestItemLoader):
            url_in = MapCompose(processor)

        it = TestItem(name='marta')
        il = ChildItemLoader(item=it)
        il.add_value('url', u'text')
        self.assertEqual(il.get_output_value('url'), ['marta'])
        il.replace_value('url', u'text2')
        self.assertEqual(il.get_output_value('url'), ['marta'])

    def test_add_value_on_unknown_field(self):
        il = TestItemLoader()
        self.assertRaises(KeyError, il.add_value, 'wrong_field', [u'lala', u'lolo'])

    def test_compose_processor(self):
        class TestItemLoader(NameItemLoader):
            name_out = Compose(lambda v: v[0], lambda v: v.title(), lambda v: v[:-1])

        il = TestItemLoader()
        il.add_value('name', [u'marta', u'other'])
        self.assertEqual(il.get_output_value('name'), u'Mart')
        item = il.load_item()
        self.assertEqual(item['name'], u'Mart')

    def test_partial_processor(self):
        def join(values, sep=None, loader_context=None, ignored=None):
            if sep is not None:
                return sep.join(values)
            elif loader_context and 'sep' in loader_context:
                return loader_context['sep'].join(values)
            else:
                return ''.join(values)

        class TestItemLoader(NameItemLoader):
            name_out = Compose(partial(join, sep='+'))
            url_out = Compose(partial(join, loader_context={'sep': '.'}))
            summary_out = Compose(partial(join, ignored='foo'))

        il = TestItemLoader()
        il.add_value('name', [u'rabbit', u'hole'])
        il.add_value('url', [u'rabbit', u'hole'])
        il.add_value('summary', [u'rabbit', u'hole'])
        item = il.load_item()
        self.assertEqual(item['name'], u'rabbit+hole')
        self.assertEqual(item['url'], u'rabbit.hole')
        self.assertEqual(item['summary'], u'rabbithole')


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



class CustomSpiderLoader(SpiderLoader):
    pass


class RedirectedMediaDownloadSpider(MediaDownloadSpider):
    name = 'redirectedmedia'

    def _process_url(self, url):
        return add_or_replace_parameter(
                    'http://localhost:8998/redirect-to',
                    'goto', url)


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



class _BaseTest(unittest.TestCase):

    storage_class = 'scrapy.extensions.httpcache.DbmCacheStorage'
    policy_class = 'scrapy.extensions.httpcache.RFC2616Policy'

    def setUp(self):
        self.yesterday = email.utils.formatdate(time.time() - 86400)
        self.today = email.utils.formatdate()
        self.tomorrow = email.utils.formatdate(time.time() + 86400)
        self.crawler = get_crawler(Spider)
        self.spider = self.crawler._create_spider('example.com')
        self.tmpdir = tempfile.mkdtemp()
        self.request = Request('http://www.example.com',
                               headers={'User-Agent': 'test'})
        self.response = Response('http://www.example.com',
                                 headers={'Content-Type': 'text/html'},
                                 body=b'test body',
                                 status=202)
        self.crawler.stats.open_spider(self.spider)

    def tearDown(self):
        self.crawler.stats.close_spider(self.spider, '')
        shutil.rmtree(self.tmpdir)

    def _get_settings(self, **new_settings):
        settings = {
            'HTTPCACHE_ENABLED': True,
            'HTTPCACHE_DIR': self.tmpdir,
            'HTTPCACHE_EXPIRATION_SECS': 1,
            'HTTPCACHE_IGNORE_HTTP_CODES': [],
            'HTTPCACHE_POLICY': self.policy_class,
            'HTTPCACHE_STORAGE': self.storage_class,
        }
        settings.update(new_settings)
        return Settings(settings)

    @contextmanager
    def _storage(self, **new_settings):
        with self._middleware(**new_settings) as mw:
            yield mw.storage

    @contextmanager
    def _policy(self, **new_settings):
        with self._middleware(**new_settings) as mw:
            yield mw.policy

    @contextmanager
    def _middleware(self, **new_settings):
        settings = self._get_settings(**new_settings)
        mw = HttpCacheMiddleware(settings, self.crawler.stats)
        mw.spider_opened(self.spider)
        try:
            yield mw
        finally:
            mw.spider_closed(self.spider)

    def assertEqualResponse(self, response1, response2):
        self.assertEqual(response1.url, response2.url)
        self.assertEqual(response1.status, response2.status)
        self.assertEqual(response1.headers, response2.headers)
        self.assertEqual(response1.body, response2.body)

    def assertEqualRequest(self, request1, request2):
        self.assertEqual(request1.url, request2.url)
        self.assertEqual(request1.headers, request2.headers)
        self.assertEqual(request1.body, request2.body)

    def assertEqualRequestButWithCacheValidators(self, request1, request2):
        self.assertEqual(request1.url, request2.url)
        assert not b'If-None-Match' in request1.headers
        assert not b'If-Modified-Since' in request1.headers
        assert any(h in request2.headers for h in (b'If-None-Match', b'If-Modified-Since'))
        self.assertEqual(request1.body, request2.body)

    def test_dont_cache(self):
        with self._middleware() as mw:
            self.request.meta['dont_cache'] = True
            mw.process_response(self.request, self.response, self.spider)
            self.assertEqual(mw.storage.retrieve_response(self.spider, self.request), None)

        with self._middleware() as mw:
            self.request.meta['dont_cache'] = False
            mw.process_response(self.request, self.response, self.spider)
            if mw.policy.should_cache_response(self.response, self.request):
                self.assertIsInstance(mw.storage.retrieve_response(self.spider, self.request), self.response.__class__)


class DefaultStorageTest(_BaseTest):

    def test_storage(self):
        with self._storage() as storage:
            request2 = self.request.copy()
            assert storage.retrieve_response(self.spider, request2) is None

            storage.store_response(self.spider, self.request, self.response)
            response2 = storage.retrieve_response(self.spider, request2)
            assert isinstance(response2, HtmlResponse)  # content-type header
            self.assertEqualResponse(self.response, response2)

            time.sleep(2)  # wait for cache to expire
            assert storage.retrieve_response(self.spider, request2) is None

    def test_storage_never_expire(self):
        with self._storage(HTTPCACHE_EXPIRATION_SECS=0) as storage:
            assert storage.retrieve_response(self.spider, self.request) is None
            storage.store_response(self.spider, self.request, self.response)
            time.sleep(0.5)  # give the chance to expire
            assert storage.retrieve_response(self.spider, self.request)


class MyGoodCrawlSpider(CrawlSpider):
    name = 'goodcrawl{0}'

    rules = (
        Rule(LinkExtractor(allow=r'/html'), callback='parse_item', follow=True),
        Rule(LinkExtractor(allow=r'/text'), follow=True),
    )

    def parse_item(self, response):
        return [scrapy.Item(), dict(foo='bar')]

    def parse(self, response):
        return [scrapy.Item(), dict(nomatch='default')]


class FailureToExcInfoTest(unittest.TestCase):

    def test_failure(self):
        try:
            0/0
        except ZeroDivisionError:
            exc_info = sys.exc_info()
            failure = Failure()

        self.assertTupleEqual(exc_info, failure_to_exc_info(failure))

    def test_non_failure(self):
        self.assertIsNone(failure_to_exc_info('test'))


class CaselessDictTest(unittest.TestCase):

    def test_init_dict(self):
        seq = {'red': 1, 'black': 3}
        d = CaselessDict(seq)
        self.assertEqual(d['red'], 1)
        self.assertEqual(d['black'], 3)

    def test_init_pair_sequence(self):
        seq = (('red', 1), ('black', 3))
        d = CaselessDict(seq)
        self.assertEqual(d['red'], 1)
        self.assertEqual(d['black'], 3)

    def test_init_mapping(self):
        class MyMapping(Mapping):
            def __init__(self, **kwargs):
                self._d = kwargs

            def __getitem__(self, key):
                return self._d[key]

            def __iter__(self):
                return iter(self._d)

            def __len__(self):
                return len(self._d)

        seq = MyMapping(red=1, black=3)
        d = CaselessDict(seq)
        self.assertEqual(d['red'], 1)
        self.assertEqual(d['black'], 3)

    def test_init_mutable_mapping(self):
        class MyMutableMapping(MutableMapping):
            def __init__(self, **kwargs):
                self._d = kwargs

            def __getitem__(self, key):
                return self._d[key]

            def __setitem__(self, key, value):
                self._d[key] = value

            def __delitem__(self, key):
                del self._d[key]

            def __iter__(self):
                return iter(self._d)

            def __len__(self):
                return len(self._d)

        seq = MyMutableMapping(red=1, black=3)
        d = CaselessDict(seq)
        self.assertEqual(d['red'], 1)
        self.assertEqual(d['black'], 3)

    def test_caseless(self):
        d = CaselessDict()
        d['key_Lower'] = 1
        self.assertEqual(d['KEy_loWer'], 1)
        self.assertEqual(d.get('KEy_loWer'), 1)

        d['KEY_LOWER'] = 3
        self.assertEqual(d['key_Lower'], 3)
        self.assertEqual(d.get('key_Lower'), 3)

    def test_delete(self):
        d = CaselessDict({'key_lower': 1})
        del d['key_LOWER']
        self.assertRaises(KeyError, d.__getitem__, 'key_LOWER')
        self.assertRaises(KeyError, d.__getitem__, 'key_lower')

    def test_getdefault(self):
        d = CaselessDict()
        self.assertEqual(d.get('c', 5), 5)
        d['c'] = 10
        self.assertEqual(d.get('c', 5), 10)

    def test_setdefault(self):
        d = CaselessDict({'a': 1, 'b': 2})

        r = d.setdefault('A', 5)
        self.assertEqual(r, 1)
        self.assertEqual(d['A'], 1)

        r = d.setdefault('c', 5)
        self.assertEqual(r, 5)
        self.assertEqual(d['C'], 5)

    def test_fromkeys(self):
        keys = ('a', 'b')

        d = CaselessDict.fromkeys(keys)
        self.assertEqual(d['A'], None)
        self.assertEqual(d['B'], None)

        d = CaselessDict.fromkeys(keys, 1)
        self.assertEqual(d['A'], 1)
        self.assertEqual(d['B'], 1)

        instance = CaselessDict()
        d = instance.fromkeys(keys)
        self.assertEqual(d['A'], None)
        self.assertEqual(d['B'], None)

        d = instance.fromkeys(keys, 1)
        self.assertEqual(d['A'], 1)
        self.assertEqual(d['B'], 1)

    def test_contains(self):
        d = CaselessDict()
        d['a'] = 1
        assert 'a' in d

    def test_pop(self):
        d = CaselessDict()
        d['a'] = 1
        self.assertEqual(d.pop('A'), 1)
        self.assertRaises(KeyError, d.pop, 'A')

    def test_normkey(self):
        class MyDict(CaselessDict):
            def normkey(self, key):
                return key.title()

        d = MyDict()
        d['key-one'] = 2
        self.assertEqual(list(d.keys()), ['Key-One'])

    def test_normvalue(self):
        class MyDict(CaselessDict):
            def normvalue(self, value):
                if value is not None:
                    return value + 1

        d = MyDict({'key': 1})
        self.assertEqual(d['key'], 2)
        self.assertEqual(d.get('key'), 2)

        d = MyDict()
        d['key'] = 1
        self.assertEqual(d['key'], 2)
        self.assertEqual(d.get('key'), 2)

        d = MyDict()
        d.setdefault('key', 1)
        self.assertEqual(d['key'], 2)
        self.assertEqual(d.get('key'), 2)

        d = MyDict()
        d.update({'key': 1})
        self.assertEqual(d['key'], 2)
        self.assertEqual(d.get('key'), 2)

        d = MyDict.fromkeys(('key',), 1)
        self.assertEqual(d['key'], 2)
        self.assertEqual(d.get('key'), 2)

    def test_copy(self):
        h1 = CaselessDict({'header1': 'value'})
        h2 = copy.copy(h1)
        self.assertEqual(h1, h2)
        self.assertEqual(h1.get('header1'), h2.get('header1'))
        assert isinstance(h2, CaselessDict)

