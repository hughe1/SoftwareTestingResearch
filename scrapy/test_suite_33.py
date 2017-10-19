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


class TestRequestMetaOriginWhenCrossOrigin(MixinOriginWhenCrossOrigin, TestRefererMiddleware):
    req_meta = {'referrer_policy': POLICY_ORIGIN_WHEN_CROSS_ORIGIN}


class MyWarning(UserWarning):
    pass


class AddHttpIfNoScheme(unittest.TestCase):

    def test_add_scheme(self):
        self.assertEqual(add_http_if_no_scheme('www.example.com'),
                                               'http://www.example.com')

    def test_without_subdomain(self):
        self.assertEqual(add_http_if_no_scheme('example.com'),
                                               'http://example.com')

    def test_path(self):
        self.assertEqual(add_http_if_no_scheme('www.example.com/some/page.html'),
                                               'http://www.example.com/some/page.html')

    def test_port(self):
        self.assertEqual(add_http_if_no_scheme('www.example.com:80'),
                                               'http://www.example.com:80')

    def test_fragment(self):
        self.assertEqual(add_http_if_no_scheme('www.example.com/some/page#frag'),
                                               'http://www.example.com/some/page#frag')

    def test_query(self):
        self.assertEqual(add_http_if_no_scheme('www.example.com/do?a=1&b=2&c=3'),
                                               'http://www.example.com/do?a=1&b=2&c=3')

    def test_username_password(self):
        self.assertEqual(add_http_if_no_scheme('username:password@www.example.com'),
                                               'http://username:password@www.example.com')

    def test_complete_url(self):
        self.assertEqual(add_http_if_no_scheme('username:password@www.example.com:80/some/page/do?a=1&b=2&c=3#frag'),
                                               'http://username:password@www.example.com:80/some/page/do?a=1&b=2&c=3#frag')

    def test_preserve_http(self):
        self.assertEqual(add_http_if_no_scheme('http://www.example.com'),
                                               'http://www.example.com')

    def test_preserve_http_without_subdomain(self):
        self.assertEqual(add_http_if_no_scheme('http://example.com'),
                                               'http://example.com')

    def test_preserve_http_path(self):
        self.assertEqual(add_http_if_no_scheme('http://www.example.com/some/page.html'),
                                               'http://www.example.com/some/page.html')

    def test_preserve_http_port(self):
        self.assertEqual(add_http_if_no_scheme('http://www.example.com:80'),
                                               'http://www.example.com:80')

    def test_preserve_http_fragment(self):
        self.assertEqual(add_http_if_no_scheme('http://www.example.com/some/page#frag'),
                                               'http://www.example.com/some/page#frag')

    def test_preserve_http_query(self):
        self.assertEqual(add_http_if_no_scheme('http://www.example.com/do?a=1&b=2&c=3'),
                                               'http://www.example.com/do?a=1&b=2&c=3')

    def test_preserve_http_username_password(self):
        self.assertEqual(add_http_if_no_scheme('http://username:password@www.example.com'),
                                               'http://username:password@www.example.com')

    def test_preserve_http_complete_url(self):
        self.assertEqual(add_http_if_no_scheme('http://username:password@www.example.com:80/some/page/do?a=1&b=2&c=3#frag'),
                                               'http://username:password@www.example.com:80/some/page/do?a=1&b=2&c=3#frag')

    def test_protocol_relative(self):
        self.assertEqual(add_http_if_no_scheme('//www.example.com'),
                                               'http://www.example.com')

    def test_protocol_relative_without_subdomain(self):
        self.assertEqual(add_http_if_no_scheme('//example.com'),
                                               'http://example.com')

    def test_protocol_relative_path(self):
        self.assertEqual(add_http_if_no_scheme('//www.example.com/some/page.html'),
                                               'http://www.example.com/some/page.html')

    def test_protocol_relative_port(self):
        self.assertEqual(add_http_if_no_scheme('//www.example.com:80'),
                                               'http://www.example.com:80')

    def test_protocol_relative_fragment(self):
        self.assertEqual(add_http_if_no_scheme('//www.example.com/some/page#frag'),
                                               'http://www.example.com/some/page#frag')

    def test_protocol_relative_query(self):
        self.assertEqual(add_http_if_no_scheme('//www.example.com/do?a=1&b=2&c=3'),
                                               'http://www.example.com/do?a=1&b=2&c=3')

    def test_protocol_relative_username_password(self):
        self.assertEqual(add_http_if_no_scheme('//username:password@www.example.com'),
                                               'http://username:password@www.example.com')

    def test_protocol_relative_complete_url(self):
        self.assertEqual(add_http_if_no_scheme('//username:password@www.example.com:80/some/page/do?a=1&b=2&c=3#frag'),
                                               'http://username:password@www.example.com:80/some/page/do?a=1&b=2&c=3#frag')

    def test_preserve_https(self):
        self.assertEqual(add_http_if_no_scheme('https://www.example.com'),
                                               'https://www.example.com')

    def test_preserve_ftp(self):
        self.assertEqual(add_http_if_no_scheme('ftp://www.example.com'),
                                               'ftp://www.example.com')


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

class HttpProxyTestCase(unittest.TestCase):
    download_handler_cls = HTTPDownloadHandler

    def setUp(self):
        site = server.Site(UriResource(), timeout=None)
        wrapper = WrappingFactory(site)
        self.port = reactor.listenTCP(0, wrapper, interface='127.0.0.1')
        self.portno = self.port.getHost().port
        self.download_handler = self.download_handler_cls(Settings())
        self.download_request = self.download_handler.download_request

    @defer.inlineCallbacks
    def tearDown(self):
        yield self.port.stopListening()
        if hasattr(self.download_handler, 'close'):
            yield self.download_handler.close()

    def getURL(self, path):
        return "http://127.0.0.1:%d/%s" % (self.portno, path)

    def test_download_with_proxy(self):
        def _test(response):
            self.assertEqual(response.status, 200)
            self.assertEqual(response.url, request.url)
            self.assertEqual(response.body, b'http://example.com')

        http_proxy = self.getURL('')
        request = Request('http://example.com', meta={'proxy': http_proxy})
        return self.download_request(request, Spider('foo')).addCallback(_test)

    def test_download_with_proxy_https_noconnect(self):
        def _test(response):
            self.assertEqual(response.status, 200)
            self.assertEqual(response.url, request.url)
            self.assertEqual(response.body, b'https://example.com')

        http_proxy = '%s?noconnect' % self.getURL('')
        request = Request('https://example.com', meta={'proxy': http_proxy})
        return self.download_request(request, Spider('foo')).addCallback(_test)

    def test_download_without_proxy(self):
        def _test(response):
            self.assertEqual(response.status, 200)
            self.assertEqual(response.url, request.url)
            self.assertEqual(response.body, b'/path/to/resource')

        request = Request(self.getURL('path/to/resource'))
        return self.download_request(request, Spider('foo')).addCallback(_test)


class UtilsRequestTest(unittest.TestCase):

    def test_request_fingerprint(self):
        r1 = Request("http://www.example.com/query?id=111&cat=222")
        r2 = Request("http://www.example.com/query?cat=222&id=111")
        self.assertEqual(request_fingerprint(r1), request_fingerprint(r1))
        self.assertEqual(request_fingerprint(r1), request_fingerprint(r2))

        r1 = Request('http://www.example.com/hnnoticiaj1.aspx?78132,199')
        r2 = Request('http://www.example.com/hnnoticiaj1.aspx?78160,199')
        self.assertNotEqual(request_fingerprint(r1), request_fingerprint(r2))

        # make sure caching is working
        self.assertEqual(request_fingerprint(r1), _fingerprint_cache[r1][None])

        r1 = Request("http://www.example.com/members/offers.html")
        r2 = Request("http://www.example.com/members/offers.html")
        r2.headers['SESSIONID'] = b"somehash"
        self.assertEqual(request_fingerprint(r1), request_fingerprint(r2))

        r1 = Request("http://www.example.com/")
        r2 = Request("http://www.example.com/")
        r2.headers['Accept-Language'] = b'en'
        r3 = Request("http://www.example.com/")
        r3.headers['Accept-Language'] = b'en'
        r3.headers['SESSIONID'] = b"somehash"

        self.assertEqual(request_fingerprint(r1), request_fingerprint(r2), request_fingerprint(r3))

        self.assertEqual(request_fingerprint(r1),
                         request_fingerprint(r1, include_headers=['Accept-Language']))

        self.assertNotEqual(request_fingerprint(r1),
                         request_fingerprint(r2, include_headers=['Accept-Language']))

        self.assertEqual(request_fingerprint(r3, include_headers=['accept-language', 'sessionid']),
                         request_fingerprint(r3, include_headers=['SESSIONID', 'Accept-Language']))

        r1 = Request("http://www.example.com")
        r2 = Request("http://www.example.com", method='POST')
        r3 = Request("http://www.example.com", method='POST', body=b'request body')

        self.assertNotEqual(request_fingerprint(r1), request_fingerprint(r2))
        self.assertNotEqual(request_fingerprint(r2), request_fingerprint(r3))

        # cached fingerprint must be cleared on request copy
        r1 = Request("http://www.example.com")
        fp1 = request_fingerprint(r1)
        r2 = r1.replace(url="http://www.example.com/other")
        fp2 = request_fingerprint(r2)
        self.assertNotEqual(fp1, fp2)

    def test_request_authenticate(self):
        r = Request("http://www.example.com")
        request_authenticate(r, 'someuser', 'somepass')
        self.assertEqual(r.headers['Authorization'], b'Basic c29tZXVzZXI6c29tZXBhc3M=')

    def test_request_httprepr(self):
        r1 = Request("http://www.example.com")
        self.assertEqual(request_httprepr(r1), b'GET / HTTP/1.1\r\nHost: www.example.com\r\n\r\n')

        r1 = Request("http://www.example.com/some/page.html?arg=1")
        self.assertEqual(request_httprepr(r1), b'GET /some/page.html?arg=1 HTTP/1.1\r\nHost: www.example.com\r\n\r\n')

        r1 = Request("http://www.example.com", method='POST', headers={"Content-type": b"text/html"}, body=b"Some body")
        self.assertEqual(request_httprepr(r1), b'POST / HTTP/1.1\r\nHost: www.example.com\r\nContent-Type: text/html\r\n\r\nSome body')

    def test_request_httprepr_for_non_http_request(self):
        # the representation is not important but it must not fail.
        request_httprepr(Request("file:///tmp/foo.txt"))
        request_httprepr(Request("ftp://localhost/tmp/foo.txt"))

if __name__ == "__main__":
    unittest.main()

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


class RegexLinkExtractorTestCase(unittest.TestCase):
    # XXX: RegexLinkExtractor is not deprecated yet, but it must be rewritten
    # not to depend on SgmlLinkExractor. Its speed is also much worse
    # than it should be.

    def setUp(self):
        body = get_testdata('link_extractor', 'sgml_linkextractor.html')
        self.response = HtmlResponse(url='http://example.com/index', body=body)

    def test_extraction(self):
        # Default arguments
        lx = RegexLinkExtractor()
        self.assertEqual(lx.extract_links(self.response),
                         [Link(url='http://example.com/sample2.html', text=u'sample 2'),
                          Link(url='http://example.com/sample3.html', text=u'sample 3 text'),
                          Link(url='http://example.com/sample3.html#foo', text=u'sample 3 repetition with fragment'),
                          Link(url='http://www.google.com/something', text=u''),
                          Link(url='http://example.com/innertag.html', text=u'inner tag'),])

    def test_link_wrong_href(self):
        html = """
        <a href="http://example.org/item1.html">Item 1</a>
        <a href="http://[example.org/item2.html">Item 2</a>
        <a href="http://example.org/item3.html">Item 3</a>
        """
        response = HtmlResponse("http://example.org/index.html", body=html)
        lx = RegexLinkExtractor()
        self.assertEqual([link for link in lx.extract_links(response)], [
            Link(url='http://example.org/item1.html', text=u'Item 1', nofollow=False),
            Link(url='http://example.org/item3.html', text=u'Item 3', nofollow=False),
        ])

    def test_html_base_href(self):
        html = """
        <html>
            <head>
                <base href="http://b.com/">
            </head>
            <body>
                <a href="test.html"></a>
            </body>
        </html>
        """
        response = HtmlResponse("http://a.com/", body=html)
        lx = RegexLinkExtractor()
        self.assertEqual([link for link in lx.extract_links(response)], [
            Link(url='http://b.com/test.html', text=u'', nofollow=False),
        ])

    @unittest.expectedFailure
    def test_extraction(self):
        # RegexLinkExtractor doesn't parse URLs with leading/trailing
        # whitespaces correctly.
        super(RegexLinkExtractorTestCase, self).test_extraction()

class DummyDH(object):

    def __init__(self, crawler):
        pass


class TestItem(Item):
    name = Field()
    age = Field()


class TestRefererMiddlewareDefault(MixinDefault, TestRefererMiddleware):
    pass


# --- Tests using settings to set policy using class path
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

class NewName(SomeBaseClass):
    pass


class TestSettingsNoReferrerWhenDowngrade(MixinNoReferrerWhenDowngrade, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.NoReferrerWhenDowngradePolicy'}


class Http10ProxyTestCase(HttpProxyTestCase):
    download_handler_cls = HTTP10DownloadHandler


class DictItemsSpider(TestSpider):
    item_cls = dict


def start_test_site(debug=False):
    root_dir = os.path.join(tests_datadir, "test_site")
    r = static.File(root_dir)
    r.putChild(b"redirect", util.Redirect(b"/redirected"))
    r.putChild(b"redirected", static.Data(b"Redirected here", "text/plain"))

    port = reactor.listenTCP(0, server.Site(r), interface="127.0.0.1")
    if debug:
        print("Test server running at http://localhost:%d/ - hit Ctrl-C to finish." \
            % port.getHost().port)
    return port


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


class SendCatchLogDeferredTest(SendCatchLogTest):

    def _get_result(self, signal, *a, **kw):
        return send_catch_log_deferred(signal, *a, **kw)


class TestOffsiteMiddleware4(TestOffsiteMiddleware3):

    def _get_spider(self):
      bad_hostname = urlparse('http:////scrapytest.org').hostname
      return dict(name='foo', allowed_domains=['scrapytest.org', None, bad_hostname])

    def test_process_spider_output(self):
      res = Response('http://scrapytest.org')
      reqs = [Request('http://scrapytest.org/1')]
      out = list(self.mw.process_spider_output(res, reqs, self.spider))
      self.assertEqual(out, reqs)

class TestRequestMetaOriginWhenCrossOrigin(MixinOriginWhenCrossOrigin, TestRefererMiddleware):
    req_meta = {'referrer_policy': POLICY_ORIGIN_WHEN_CROSS_ORIGIN}


class LoadTestCase(unittest.TestCase):

    def test_enabled_handler(self):
        handlers = {'scheme': 'tests.test_downloader_handlers.DummyDH'}
        crawler = get_crawler(settings_dict={'DOWNLOAD_HANDLERS': handlers})
        dh = DownloadHandlers(crawler)
        self.assertIn('scheme', dh._schemes)
        for scheme in handlers: # force load handlers
            dh._get_handler(scheme)
        self.assertIn('scheme', dh._handlers)
        self.assertNotIn('scheme', dh._notconfigured)

    def test_not_configured_handler(self):
        handlers = {'scheme': 'tests.test_downloader_handlers.OffDH'}
        crawler = get_crawler(settings_dict={'DOWNLOAD_HANDLERS': handlers})
        dh = DownloadHandlers(crawler)
        self.assertIn('scheme', dh._schemes)
        for scheme in handlers: # force load handlers
            dh._get_handler(scheme)
        self.assertNotIn('scheme', dh._handlers)
        self.assertIn('scheme', dh._notconfigured)

    def test_disabled_handler(self):
        handlers = {'scheme': None}
        crawler = get_crawler(settings_dict={'DOWNLOAD_HANDLERS': handlers})
        dh = DownloadHandlers(crawler)
        self.assertNotIn('scheme', dh._schemes)
        for scheme in handlers: # force load handlers
            dh._get_handler(scheme)
        self.assertNotIn('scheme', dh._handlers)
        self.assertIn('scheme', dh._notconfigured)


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


class NestedItemLoader(ItemLoader):
    default_item_class = TestNestedItem


@contextlib.contextmanager
def inside_a_project():
    prev_dir = os.getcwd()
    project_dir = tempfile.mkdtemp()

    try:
        os.chdir(project_dir)
        with open('scrapy.cfg', 'w') as f:
            # create an empty scrapy.cfg
            f.close()

        yield project_dir
    finally:
        os.chdir(prev_dir)
        shutil.rmtree(project_dir)


class DownloadTimeoutMiddlewareTest(unittest.TestCase):

    def get_request_spider_mw(self, settings=None):
        crawler = get_crawler(Spider, settings)
        spider = crawler._create_spider('foo')
        request = Request('http://scrapytest.org/')
        return request, spider, DownloadTimeoutMiddleware.from_crawler(crawler)

    def test_default_download_timeout(self):
        req, spider, mw = self.get_request_spider_mw()
        mw.spider_opened(spider)
        assert mw.process_request(req, spider) is None
        self.assertEqual(req.meta.get('download_timeout'), 180)

    def test_string_download_timeout(self):
        req, spider, mw = self.get_request_spider_mw({'DOWNLOAD_TIMEOUT': '20.1'})
        mw.spider_opened(spider)
        assert mw.process_request(req, spider) is None
        self.assertEqual(req.meta.get('download_timeout'), 20.1)

    def test_spider_has_download_timeout(self):
        req, spider, mw = self.get_request_spider_mw()
        spider.download_timeout = 2
        mw.spider_opened(spider)
        assert mw.process_request(req, spider) is None
        self.assertEqual(req.meta.get('download_timeout'), 2)

    def test_request_has_download_timeout(self):
        req, spider, mw = self.get_request_spider_mw()
        spider.download_timeout = 2
        mw.spider_opened(spider)
        req.meta['download_timeout'] = 1
        assert mw.process_request(req, spider) is None
        self.assertEqual(req.meta.get('download_timeout'), 1)
