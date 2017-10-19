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



class TestRequestMetaOriginWhenCrossOrigin(MixinOriginWhenCrossOrigin, TestRefererMiddleware):
    req_meta = {'referrer_policy': POLICY_ORIGIN_WHEN_CROSS_ORIGIN}


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

class EncodingResource(resource.Resource):
    out_encoding = 'cp1251'

    def render(self, request):
        body = to_unicode(request.content.read())
        request.setHeader(b'content-encoding', self.out_encoding)
        return body.encode(self.out_encoding)


class Http10TestCase(HttpTestCase):
    """HTTP 1.0 test case"""
    download_handler_cls = HTTP10DownloadHandler


class TestRequestMetaStrictOriginWhenCrossOrigin(MixinStrictOriginWhenCrossOrigin, TestRefererMiddleware):
    req_meta = {'referrer_policy': POLICY_STRICT_ORIGIN_WHEN_CROSS_ORIGIN}


class TestHttpErrorMiddlewareSettings(TestCase):
    """Similar test, but with settings"""

    def setUp(self):
        self.spider = Spider('foo')
        self.mw = HttpErrorMiddleware(Settings({'HTTPERROR_ALLOWED_CODES': (402,)}))
        self.req = Request('http://scrapytest.org')
        self.res200, self.res404, self.res402 = _responses(self.req, [200, 404, 402])

    def test_process_spider_input(self):
        self.assertEqual(None,
                self.mw.process_spider_input(self.res200, self.spider))
        self.assertRaises(HttpError,
                self.mw.process_spider_input, self.res404, self.spider)
        self.assertEqual(None,
                self.mw.process_spider_input(self.res402, self.spider))

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

    def test_spider_override_settings(self):
        self.spider.handle_httpstatus_list = [404]
        self.assertEqual(None,
            self.mw.process_spider_input(self.res404, self.spider))
        self.assertRaises(HttpError,
                self.mw.process_spider_input, self.res402, self.spider)


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

class StartprojectTest(ProjectTest):

    def test_startproject(self):
        self.assertEqual(0, self.call('startproject', self.project_name))

        assert exists(join(self.proj_path, 'scrapy.cfg'))
        assert exists(join(self.proj_path, 'testproject'))
        assert exists(join(self.proj_mod_path, '__init__.py'))
        assert exists(join(self.proj_mod_path, 'items.py'))
        assert exists(join(self.proj_mod_path, 'pipelines.py'))
        assert exists(join(self.proj_mod_path, 'settings.py'))
        assert exists(join(self.proj_mod_path, 'spiders', '__init__.py'))

        self.assertEqual(1, self.call('startproject', self.project_name))
        self.assertEqual(1, self.call('startproject', 'wrong---project---name'))
        self.assertEqual(1, self.call('startproject', 'sys'))

    def test_startproject_with_project_dir(self):
        project_dir = mkdtemp()
        self.assertEqual(0, self.call('startproject', self.project_name, project_dir))

        assert exists(join(abspath(project_dir), 'scrapy.cfg'))
        assert exists(join(abspath(project_dir), 'testproject'))
        assert exists(join(join(abspath(project_dir), self.project_name), '__init__.py'))
        assert exists(join(join(abspath(project_dir), self.project_name), 'items.py'))
        assert exists(join(join(abspath(project_dir), self.project_name), 'pipelines.py'))
        assert exists(join(join(abspath(project_dir), self.project_name), 'settings.py'))
        assert exists(join(join(abspath(project_dir), self.project_name), 'spiders', '__init__.py'))

        self.assertEqual(0, self.call('startproject', self.project_name, project_dir + '2'))

        self.assertEqual(1, self.call('startproject', self.project_name, project_dir))
        self.assertEqual(1, self.call('startproject', self.project_name + '2', project_dir))
        self.assertEqual(1, self.call('startproject', 'wrong---project---name'))
        self.assertEqual(1, self.call('startproject', 'sys'))
        self.assertEqual(2, self.call('startproject'))
        self.assertEqual(2, self.call('startproject', self.project_name, project_dir, 'another_params'))


class RFC2616PolicyTest(DefaultStorageTest):

    policy_class = 'scrapy.extensions.httpcache.RFC2616Policy'

    def _process_requestresponse(self, mw, request, response):
        result = None
        try:
            result = mw.process_request(request, self.spider)
            if result:
                assert isinstance(result, (Request, Response))
                return result
            else:
                result = mw.process_response(request, response, self.spider)
                assert isinstance(result, Response)
                return result
        except Exception:
            print('Request', request)
            print('Response', response)
            print('Result', result)
            raise

    def test_request_cacheability(self):
        res0 = Response(self.request.url, status=200,
                        headers={'Expires': self.tomorrow})
        req0 = Request('http://example.com')
        req1 = req0.replace(headers={'Cache-Control': 'no-store'})
        req2 = req0.replace(headers={'Cache-Control': 'no-cache'})
        with self._middleware() as mw:
            # response for a request with no-store must not be cached
            res1 = self._process_requestresponse(mw, req1, res0)
            self.assertEqualResponse(res1, res0)
            assert mw.storage.retrieve_response(self.spider, req1) is None
            # Re-do request without no-store and expect it to be cached
            res2 = self._process_requestresponse(mw, req0, res0)
            assert 'cached' not in res2.flags
            res3 = mw.process_request(req0, self.spider)
            assert 'cached' in res3.flags
            self.assertEqualResponse(res2, res3)
            # request with no-cache directive must not return cached response
            # but it allows new response to be stored
            res0b = res0.replace(body=b'foo')
            res4 = self._process_requestresponse(mw, req2, res0b)
            self.assertEqualResponse(res4, res0b)
            assert 'cached' not in res4.flags
            res5 = self._process_requestresponse(mw, req0, None)
            self.assertEqualResponse(res5, res0b)
            assert 'cached' in res5.flags

    def test_response_cacheability(self):
        responses = [
            # 304 is not cacheable no matter what servers sends
            (False, 304, {}),
            (False, 304, {'Last-Modified': self.yesterday}),
            (False, 304, {'Expires': self.tomorrow}),
            (False, 304, {'Etag': 'bar'}),
            (False, 304, {'Cache-Control': 'max-age=3600'}),
            # Always obey no-store cache control
            (False, 200, {'Cache-Control': 'no-store'}),
            (False, 200, {'Cache-Control': 'no-store, max-age=300'}),  # invalid
            (False, 200, {'Cache-Control': 'no-store', 'Expires': self.tomorrow}),  # invalid
            # Ignore responses missing expiration and/or validation headers
            (False, 200, {}),
            (False, 302, {}),
            (False, 307, {}),
            (False, 404, {}),
            # Cache responses with expiration and/or validation headers
            (True, 200, {'Last-Modified': self.yesterday}),
            (True, 203, {'Last-Modified': self.yesterday}),
            (True, 300, {'Last-Modified': self.yesterday}),
            (True, 301, {'Last-Modified': self.yesterday}),
            (True, 308, {'Last-Modified': self.yesterday}),
            (True, 401, {'Last-Modified': self.yesterday}),
            (True, 404, {'Cache-Control': 'public, max-age=600'}),
            (True, 302, {'Expires': self.tomorrow}),
            (True, 200, {'Etag': 'foo'}),
        ]
        with self._middleware() as mw:
            for idx, (shouldcache, status, headers) in enumerate(responses):
                req0 = Request('http://example-%d.com' % idx)
                res0 = Response(req0.url, status=status, headers=headers)
                res1 = self._process_requestresponse(mw, req0, res0)
                res304 = res0.replace(status=304)
                res2 = self._process_requestresponse(mw, req0, res304 if shouldcache else res0)
                self.assertEqualResponse(res1, res0)
                self.assertEqualResponse(res2, res0)
                resc = mw.storage.retrieve_response(self.spider, req0)
                if shouldcache:
                    self.assertEqualResponse(resc, res1)
                    assert 'cached' in res2.flags and res2.status != 304
                else:
                    self.assertFalse(resc)
                    assert 'cached' not in res2.flags

        # cache unconditionally unless response contains no-store or is a 304
        with self._middleware(HTTPCACHE_ALWAYS_STORE=True) as mw:
            for idx, (_, status, headers) in enumerate(responses):
                shouldcache = 'no-store' not in headers.get('Cache-Control', '') and status != 304
                req0 = Request('http://example2-%d.com' % idx)
                res0 = Response(req0.url, status=status, headers=headers)
                res1 = self._process_requestresponse(mw, req0, res0)
                res304 = res0.replace(status=304)
                res2 = self._process_requestresponse(mw, req0, res304 if shouldcache else res0)
                self.assertEqualResponse(res1, res0)
                self.assertEqualResponse(res2, res0)
                resc = mw.storage.retrieve_response(self.spider, req0)
                if shouldcache:
                    self.assertEqualResponse(resc, res1)
                    assert 'cached' in res2.flags and res2.status != 304
                else:
                    self.assertFalse(resc)
                    assert 'cached' not in res2.flags

    def test_cached_and_fresh(self):
        sampledata = [
            (200, {'Date': self.yesterday, 'Expires': self.tomorrow}),
            (200, {'Date': self.yesterday, 'Cache-Control': 'max-age=86405'}),
            (200, {'Age': '299', 'Cache-Control': 'max-age=300'}),
            # Obey max-age if present over any others
            (200, {'Date': self.today,
                   'Age': '86405',
                   'Cache-Control': 'max-age=' + str(86400 * 3),
                   'Expires': self.yesterday,
                   'Last-Modified': self.yesterday,
                   }),
            # obey Expires if max-age is not present
            (200, {'Date': self.yesterday,
                   'Age': '86400',
                   'Cache-Control': 'public',
                   'Expires': self.tomorrow,
                   'Last-Modified': self.yesterday,
                   }),
            # Default missing Date header to right now
            (200, {'Expires': self.tomorrow}),
            # Firefox - Expires if age is greater than 10% of (Date - Last-Modified)
            (200, {'Date': self.today, 'Last-Modified': self.yesterday, 'Age': str(86400 / 10 - 1)}),
            # Firefox - Set one year maxage to permanent redirects missing expiration info
            (300, {}), (301, {}), (308, {}),
        ]
        with self._middleware() as mw:
            for idx, (status, headers) in enumerate(sampledata):
                req0 = Request('http://example-%d.com' % idx)
                res0 = Response(req0.url, status=status, headers=headers)
                # cache fresh response
                res1 = self._process_requestresponse(mw, req0, res0)
                self.assertEqualResponse(res1, res0)
                assert 'cached' not in res1.flags
                # return fresh cached response without network interaction
                res2 = self._process_requestresponse(mw, req0, None)
                self.assertEqualResponse(res1, res2)
                assert 'cached' in res2.flags
                # validate cached response if request max-age set as 0
                req1 = req0.replace(headers={'Cache-Control': 'max-age=0'})
                res304 = res0.replace(status=304)
                assert mw.process_request(req1, self.spider) is None
                res3 = self._process_requestresponse(mw, req1, res304)
                self.assertEqualResponse(res1, res3)
                assert 'cached' in res3.flags

    def test_cached_and_stale(self):
        sampledata = [
            (200, {'Date': self.today, 'Expires': self.yesterday}),
            (200, {'Date': self.today, 'Expires': self.yesterday, 'Last-Modified': self.yesterday}),
            (200, {'Expires': self.yesterday}),
            (200, {'Expires': self.yesterday, 'ETag': 'foo'}),
            (200, {'Expires': self.yesterday, 'Last-Modified': self.yesterday}),
            (200, {'Expires': self.tomorrow, 'Age': '86405'}),
            (200, {'Cache-Control': 'max-age=86400', 'Age': '86405'}),
            # no-cache forces expiration, also revalidation if validators exists
            (200, {'Cache-Control': 'no-cache'}),
            (200, {'Cache-Control': 'no-cache', 'ETag': 'foo'}),
            (200, {'Cache-Control': 'no-cache', 'Last-Modified': self.yesterday}),
            (200, {'Cache-Control': 'no-cache,must-revalidate', 'Last-Modified': self.yesterday}),
            (200, {'Cache-Control': 'must-revalidate', 'Expires': self.yesterday, 'Last-Modified': self.yesterday}),
            (200, {'Cache-Control': 'max-age=86400,must-revalidate', 'Age': '86405'}),
        ]
        with self._middleware() as mw:
            for idx, (status, headers) in enumerate(sampledata):
                req0 = Request('http://example-%d.com' % idx)
                res0a = Response(req0.url, status=status, headers=headers)
                # cache expired response
                res1 = self._process_requestresponse(mw, req0, res0a)
                self.assertEqualResponse(res1, res0a)
                assert 'cached' not in res1.flags
                # Same request but as cached response is stale a new response must
                # be returned
                res0b = res0a.replace(body=b'bar')
                res2 = self._process_requestresponse(mw, req0, res0b)
                self.assertEqualResponse(res2, res0b)
                assert 'cached' not in res2.flags
                cc = headers.get('Cache-Control', '')
                # Previous response expired too, subsequent request to same
                # resource must revalidate and succeed on 304 if validators
                # are present
                if 'ETag' in headers or 'Last-Modified' in headers:
                    res0c = res0b.replace(status=304)
                    res3 = self._process_requestresponse(mw, req0, res0c)
                    self.assertEqualResponse(res3, res0b)
                    assert 'cached' in res3.flags
                    # get cached response on server errors unless must-revalidate
                    # in cached response
                    res0d = res0b.replace(status=500)
                    res4 = self._process_requestresponse(mw, req0, res0d)
                    if 'must-revalidate' in cc:
                        assert 'cached' not in res4.flags
                        self.assertEqualResponse(res4, res0d)
                    else:
                        assert 'cached' in res4.flags
                        self.assertEqualResponse(res4, res0b)
                # Requests with max-stale can fetch expired cached responses
                # unless cached response has must-revalidate
                req1 = req0.replace(headers={'Cache-Control': 'max-stale'})
                res5 = self._process_requestresponse(mw, req1, res0b)
                self.assertEqualResponse(res5, res0b)
                if 'no-cache' in cc or 'must-revalidate' in cc:
                    assert 'cached' not in res5.flags
                else:
                    assert 'cached' in res5.flags

    def test_process_exception(self):
        with self._middleware() as mw:
            res0 = Response(self.request.url, headers={'Expires': self.yesterday})
            req0 = Request(self.request.url)
            self._process_requestresponse(mw, req0, res0)
            for e in mw.DOWNLOAD_EXCEPTIONS:
                # Simulate encountering an error on download attempts
                assert mw.process_request(req0, self.spider) is None
                res1 = mw.process_exception(req0, e('foo'), self.spider)
                # Use cached response as recovery
                assert 'cached' in res1.flags
                self.assertEqualResponse(res0, res1)
            # Do not use cached response for unhandled exceptions
            mw.process_request(req0, self.spider)
            assert mw.process_exception(req0, Exception('foo'), self.spider) is None

    def test_ignore_response_cache_controls(self):
        sampledata = [
            (200, {'Date': self.yesterday, 'Expires': self.tomorrow}),
            (200, {'Date': self.yesterday, 'Cache-Control': 'no-store,max-age=86405'}),
            (200, {'Age': '299', 'Cache-Control': 'max-age=300,no-cache'}),
            (300, {'Cache-Control': 'no-cache'}),
            (200, {'Expires': self.tomorrow, 'Cache-Control': 'no-store'}),
        ]
        with self._middleware(HTTPCACHE_IGNORE_RESPONSE_CACHE_CONTROLS=['no-cache', 'no-store']) as mw:
            for idx, (status, headers) in enumerate(sampledata):
                req0 = Request('http://example-%d.com' % idx)
                res0 = Response(req0.url, status=status, headers=headers)
                # cache fresh response
                res1 = self._process_requestresponse(mw, req0, res0)
                self.assertEqualResponse(res1, res0)
                assert 'cached' not in res1.flags
                # return fresh cached response without network interaction
                res2 = self._process_requestresponse(mw, req0, None)
                self.assertEqualResponse(res1, res2)
                assert 'cached' in res2.flags

if __name__ == '__main__':
    unittest.main()

class SpiderStateTest(unittest.TestCase):

    def test_store_load(self):
        jobdir = self.mktemp()
        os.mkdir(jobdir)
        try:
            spider = Spider(name='default')
            dt = datetime.now()

            ss = SpiderState(jobdir)
            ss.spider_opened(spider)
            spider.state['one'] = 1
            spider.state['dt'] = dt
            ss.spider_closed(spider)

            spider2 = Spider(name='default')
            ss2 = SpiderState(jobdir)
            ss2.spider_opened(spider2)
            self.assertEqual(spider.state, {'one': 1, 'dt': dt})
            ss2.spider_closed(spider2)
        finally:
            shutil.rmtree(jobdir)

    def test_state_attribute(self):
        # state attribute must be present if jobdir is not set, to provide a
        # consistent interface
        spider = Spider(name='default')
        ss = SpiderState()
        ss.spider_opened(spider)
        self.assertEqual(spider.state, {})
        ss.spider_closed(spider)

    def test_not_configured(self):
        crawler = get_crawler(Spider)
        self.assertRaises(NotConfigured, SpiderState.from_crawler, crawler)

class LinkTest(unittest.TestCase):

    def _assert_same_links(self, link1, link2):
        self.assertEqual(link1, link2)
        self.assertEqual(hash(link1), hash(link2))

    def _assert_different_links(self, link1, link2):
        self.assertNotEqual(link1, link2)
        self.assertNotEqual(hash(link1), hash(link2))

    def test_eq_and_hash(self):
        l1 = Link("http://www.example.com")
        l2 = Link("http://www.example.com/other")
        l3 = Link("http://www.example.com")

        self._assert_same_links(l1, l1)
        self._assert_different_links(l1, l2)
        self._assert_same_links(l1, l3)

        l4 = Link("http://www.example.com", text="test")
        l5 = Link("http://www.example.com", text="test2")
        l6 = Link("http://www.example.com", text="test")

        self._assert_same_links(l4, l4)
        self._assert_different_links(l4, l5)
        self._assert_same_links(l4, l6)

        l7 = Link("http://www.example.com", text="test", fragment='something', nofollow=False)
        l8 = Link("http://www.example.com", text="test", fragment='something', nofollow=False)
        l9 = Link("http://www.example.com", text="test", fragment='something', nofollow=True)
        l10 = Link("http://www.example.com", text="test", fragment='other', nofollow=False)
        self._assert_same_links(l7, l8)
        self._assert_different_links(l7, l9)
        self._assert_different_links(l7, l10)

    def test_repr(self):
        l1 = Link("http://www.example.com", text="test", fragment='something', nofollow=True)
        l2 = eval(repr(l1))
        self._assert_same_links(l1, l2)

    def test_non_str_url_py2(self):
        if six.PY2:
            with warnings.catch_warnings(record=True) as w:
                link = Link(u"http://www.example.com/\xa3")
                self.assertIsInstance(link.url, str)
                self.assertEqual(link.url, b'http://www.example.com/\xc2\xa3')
            assert len(w) == 1, "warning not issued"
        else:
            with self.assertRaises(TypeError):
                Link(b"http://www.example.com/\xc2\xa3")

class Http11MockServerTestCase(unittest.TestCase):
    """HTTP 1.1 test case with MockServer"""

    def setUp(self):
        self.mockserver = MockServer()
        self.mockserver.__enter__()

    def tearDown(self):
        self.mockserver.__exit__(None, None, None)

    @defer.inlineCallbacks
    def test_download_with_content_length(self):
        crawler = get_crawler(SingleRequestSpider)
        # http://localhost:8998/partial set Content-Length to 1024, use download_maxsize= 1000 to avoid
        # download it
        yield crawler.crawl(seed=Request(url='http://localhost:8998/partial', meta={'download_maxsize': 1000}))
        failure = crawler.spider.meta['failure']
        self.assertIsInstance(failure.value, defer.CancelledError)

    @defer.inlineCallbacks
    def test_download(self):
        crawler = get_crawler(SingleRequestSpider)
        yield crawler.crawl(seed=Request(url='http://localhost:8998'))
        failure = crawler.spider.meta.get('failure')
        self.assertTrue(failure == None)
        reason = crawler.spider.meta['close_reason']
        self.assertTrue(reason, 'finished')

    @defer.inlineCallbacks
    def test_download_gzip_response(self):
        crawler = get_crawler(SingleRequestSpider)
        body = b'1' * 100  # PayloadResource requires body length to be 100
        request = Request('http://localhost:8998/payload', method='POST',
                          body=body, meta={'download_maxsize': 50})
        yield crawler.crawl(seed=request)
        failure = crawler.spider.meta['failure']
        # download_maxsize < 100, hence the CancelledError
        self.assertIsInstance(failure.value, defer.CancelledError)

        if six.PY2:
            request.headers.setdefault(b'Accept-Encoding', b'gzip,deflate')
            request = request.replace(url='http://localhost:8998/xpayload')
            yield crawler.crawl(seed=request)
            # download_maxsize = 50 is enough for the gzipped response
            failure = crawler.spider.meta.get('failure')
            self.assertTrue(failure == None)
            reason = crawler.spider.meta['close_reason']
            self.assertTrue(reason, 'finished')
        else:
            # See issue https://twistedmatrix.com/trac/ticket/8175
            raise unittest.SkipTest("xpayload only enabled for PY2")


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


class ContractsManagerTest(unittest.TestCase):
    contracts = [UrlContract, ReturnsContract, ScrapesContract]

    def setUp(self):
        self.conman = ContractsManager(self.contracts)
        self.results = TextTestResult(stream=None, descriptions=False, verbosity=0)

    def should_succeed(self):
        self.assertFalse(self.results.failures)
        self.assertFalse(self.results.errors)

    def should_fail(self):
        self.assertTrue(self.results.failures)
        self.assertFalse(self.results.errors)

    def test_contracts(self):
        spider = TestSpider()

        # extract contracts correctly
        contracts = self.conman.extract_contracts(spider.returns_request)
        self.assertEqual(len(contracts), 2)
        self.assertEqual(frozenset(type(x) for x in contracts),
            frozenset([UrlContract, ReturnsContract]))

        # returns request for valid method
        request = self.conman.from_method(spider.returns_request, self.results)
        self.assertNotEqual(request, None)

        # no request for missing url
        request = self.conman.from_method(spider.parse_no_url, self.results)
        self.assertEqual(request, None)

    def test_returns(self):
        spider = TestSpider()
        response = ResponseMock()

        # returns_item
        request = self.conman.from_method(spider.returns_item, self.results)
        request.callback(response)
        self.should_succeed()

        # returns_dict_item
        request = self.conman.from_method(spider.returns_dict_item, self.results)
        request.callback(response)
        self.should_succeed()

        # returns_request
        request = self.conman.from_method(spider.returns_request, self.results)
        request.callback(response)
        self.should_succeed()

        # returns_fail
        request = self.conman.from_method(spider.returns_fail, self.results)
        request.callback(response)
        self.should_fail()

        # returns_dict_fail
        request = self.conman.from_method(spider.returns_dict_fail, self.results)
        request.callback(response)
        self.should_fail()

    def test_scrapes(self):
        spider = TestSpider()
        response = ResponseMock()

        # scrapes_item_ok
        request = self.conman.from_method(spider.scrapes_item_ok, self.results)
        request.callback(response)
        self.should_succeed()

        # scrapes_dict_item_ok
        request = self.conman.from_method(spider.scrapes_dict_item_ok, self.results)
        request.callback(response)
        self.should_succeed()

        # scrapes_item_fail
        request = self.conman.from_method(spider.scrapes_item_fail,
                self.results)
        request.callback(response)
        self.should_fail()

        # scrapes_dict_item_fail
        request = self.conman.from_method(spider.scrapes_dict_item_fail,
                self.results)
        request.callback(response)
        self.should_fail()

class CrawlerRunnerTestCase(BaseCrawlerTest):

    def test_spider_manager_verify_interface(self):
        settings = Settings({
            'SPIDER_LOADER_CLASS': 'tests.test_crawler.SpiderLoaderWithWrongInterface'
        })
        with warnings.catch_warnings(record=True) as w, \
                self.assertRaises(AttributeError):
            CrawlerRunner(settings)
            self.assertEqual(len(w), 1)
            self.assertIn("SPIDER_LOADER_CLASS", str(w[0].message))
            self.assertIn("scrapy.interfaces.ISpiderLoader", str(w[0].message))

    def test_crawler_runner_accepts_dict(self):
        runner = CrawlerRunner({'foo': 'bar'})
        self.assertEqual(runner.settings['foo'], 'bar')
        self.assertOptionIsDefault(runner.settings, 'RETRY_ENABLED')

    def test_crawler_runner_accepts_None(self):
        runner = CrawlerRunner()
        self.assertOptionIsDefault(runner.settings, 'RETRY_ENABLED')

    def test_deprecated_attribute_spiders(self):
        with warnings.catch_warnings(record=True) as w:
            runner = CrawlerRunner(Settings())
            spiders = runner.spiders
            self.assertEqual(len(w), 1)
            self.assertIn("CrawlerRunner.spiders", str(w[0].message))
            self.assertIn("CrawlerRunner.spider_loader", str(w[0].message))
            sl_cls = load_object(runner.settings['SPIDER_LOADER_CLASS'])
            self.assertIsInstance(spiders, sl_cls)

    def test_spidermanager_deprecation(self):
        with warnings.catch_warnings(record=True) as w:
            runner = CrawlerRunner({
                'SPIDER_MANAGER_CLASS': 'tests.test_crawler.CustomSpiderLoader'
            })
            self.assertIsInstance(runner.spider_loader, CustomSpiderLoader)
            self.assertEqual(len(w), 1)
            self.assertIn('Please use SPIDER_LOADER_CLASS', str(w[0].message))


class Http11ProxyTestCase(HttpProxyTestCase):
    download_handler_cls = HTTP11DownloadHandler

    @defer.inlineCallbacks
    def test_download_with_proxy_https_timeout(self):
        """ Test TunnelingTCP4ClientEndpoint """
        http_proxy = self.getURL('')
        domain = 'https://no-such-domain.nosuch'
        request = Request(
            domain, meta={'proxy': http_proxy, 'download_timeout': 0.2})
        d = self.download_request(request, Spider('foo'))
        timeout = yield self.assertFailure(d, error.TimeoutError)
        self.assertIn(domain, timeout.osError)


class DbmStorageWithCustomDbmModuleTest(DbmStorageTest):

    dbm_module = 'tests.mocks.dummydbm'

    def _get_settings(self, **new_settings):
        new_settings.setdefault('HTTPCACHE_DBM_MODULE', self.dbm_module)
        return super(DbmStorageWithCustomDbmModuleTest, self)._get_settings(**new_settings)

    def test_custom_dbm_module_loaded(self):
        # make sure our dbm module has been loaded
        with self._storage() as storage:
            self.assertEqual(storage.dbmodule.__name__, self.dbm_module)


class TestSpider(Spider):
    name = 'test'

    def parse_item(self, response):
        pass

    def handle_error(self, failure):
        pass


class FetchTest(ProcessTest, SiteTest, unittest.TestCase):

    command = 'fetch'

    @defer.inlineCallbacks
    def test_output(self):
        _, out, _ = yield self.execute([self.url('/text')])
        self.assertEqual(out.strip(), b'Works')

    @defer.inlineCallbacks
    def test_redirect_default(self):
        _, out, _ = yield self.execute([self.url('/redirect')])
        self.assertEqual(out.strip(), b'Redirected here')

    @defer.inlineCallbacks
    def test_redirect_disabled(self):
        _, out, err = yield self.execute(['--no-redirect', self.url('/redirect-no-meta-refresh')])
        err = err.strip()
        self.assertIn(b'downloader/response_status_count/302', err, err)
        self.assertNotIn(b'downloader/response_status_count/200', err, err)

    @defer.inlineCallbacks
    def test_headers(self):
        _, out, _ = yield self.execute([self.url('/text'), '--headers'])
        out = out.replace(b'\r', b'') # required on win32
        assert b'Server: TwistedWeb' in out, out
        assert b'Content-Type: text/plain' in out

class TestReferrerOnRedirectNoReferrer(TestReferrerOnRedirect):
    """
    No Referrer policy never sets the "Referer" header.
    HTTP redirections should not change that.
    """
    settings = {'REFERRER_POLICY': 'no-referrer'}
    scenarii = [
        (   'http://scrapytest.org/1',      # parent
            'http://scrapytest.org/2',      # target
            (
                # redirections: code, URL
                (301, 'http://scrapytest.org/3'),
                (301, 'http://scrapytest.org/4'),
            ),
            None, # expected initial "Referer"
            None, # expected "Referer" for the redirection request
        ),
        (   'https://scrapytest.org/1',
            'https://scrapytest.org/2',
            (
                (301, 'http://scrapytest.org/3'),
            ),
            None,
            None,
        ),
        (   'https://scrapytest.org/1',
            'https://example.com/2',    # different origin
            (
                (301, 'http://scrapytest.com/3'),
            ),
            None,
            None,
        ),
    ]


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

class RunSpiderCommandTest(CommandTest):

    debug_log_spider = """
import scrapy

class VersionTest(ProcessTest, unittest.TestCase):

    command = 'version'

    @defer.inlineCallbacks
    def test_output(self):
        encoding = getattr(sys.stdout, 'encoding') or 'utf-8'
        _, out, _ = yield self.execute([])
        self.assertEqual(
            out.strip().decode(encoding),
            "Scrapy %s" % scrapy.__version__,
        )

    @defer.inlineCallbacks
    def test_verbose_output(self):
        encoding = getattr(sys.stdout, 'encoding') or 'utf-8'
        _, out, _ = yield self.execute(['-v'])
        headers = [l.partition(":")[0].strip()
                   for l in out.strip().decode(encoding).splitlines()]
        self.assertEqual(headers, ['Scrapy', 'lxml', 'libxml2',
                                   'cssselect', 'parsel', 'w3lib',
                                   'Twisted', 'Python', 'pyOpenSSL',
                                   'cryptography', 'Platform'])

def _test_data(formats):
    uncompressed_body = get_testdata('compressed', 'feed-sample1.xml')
    test_responses = {}
    for format in formats:
        body = get_testdata('compressed', 'feed-sample1.' + format)
        test_responses[format] = Response('http://foo.com/bar', body=body)
    return uncompressed_body, test_responses


class TestOffsiteMiddleware(TestCase):

    def setUp(self):
        crawler = get_crawler(Spider)
        self.spider = crawler._create_spider(**self._get_spiderargs())
        self.mw = OffsiteMiddleware.from_crawler(crawler)
        self.mw.spider_opened(self.spider)

    def _get_spiderargs(self):
        return dict(name='foo', allowed_domains=['scrapytest.org', 'scrapy.org', 'scrapy.test.org'])

    def test_process_spider_output(self):
        res = Response('http://scrapytest.org')

        onsite_reqs = [Request('http://scrapytest.org/1'),
                       Request('http://scrapy.org/1'),
                       Request('http://sub.scrapy.org/1'),
                       Request('http://offsite.tld/letmepass', dont_filter=True),
                       Request('http://scrapy.test.org/')]
        offsite_reqs = [Request('http://scrapy2.org'),
                       Request('http://offsite.tld/'),
                       Request('http://offsite.tld/scrapytest.org'),
                       Request('http://offsite.tld/rogue.scrapytest.org'),
                       Request('http://rogue.scrapytest.org.haha.com'),
                       Request('http://roguescrapytest.org'),
                       Request('http://test.org/'),
                       Request('http://notscrapy.test.org/')]
        reqs = onsite_reqs + offsite_reqs

        out = list(self.mw.process_spider_output(res, reqs, self.spider))
        self.assertEqual(out, onsite_reqs)

