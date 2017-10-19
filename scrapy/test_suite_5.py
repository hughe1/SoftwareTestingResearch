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


class TestRequestMetaPredecence003(MixinUnsafeUrl, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.OriginWhenCrossOriginPolicy'}
    req_meta = {'referrer_policy': POLICY_UNSAFE_URL}


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

class Bar(trackref.object_ref):
    pass


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

class GuessSchemeTest(unittest.TestCase):
    pass

def create_guess_scheme_t(args):
    def do_expected(self):
        url = guess_scheme(args[0])
        assert url.startswith(args[1]), \
            'Wrong scheme guessed: for `%s` got `%s`, expected `%s...`' % (
                args[0], url, args[1])
    return do_expected

def create_skipped_scheme_t(args):
    def do_expected(self):
        raise unittest.SkipTest(args[2])
        url = guess_scheme(args[0])
        assert url.startswith(args[1])
    return do_expected

for k, args in enumerate ([
            ('/index',                              'file://'),
            ('/index.html',                         'file://'),
            ('./index.html',                        'file://'),
            ('../index.html',                       'file://'),
            ('../../index.html',                    'file://'),
            ('./data/index.html',                   'file://'),
            ('.hidden/data/index.html',             'file://'),
            ('/home/user/www/index.html',           'file://'),
            ('//home/user/www/index.html',          'file://'),
            ('file:///home/user/www/index.html',    'file://'),

            ('index.html',                          'http://'),
            ('example.com',                         'http://'),
            ('www.example.com',                     'http://'),
            ('www.example.com/index.html',          'http://'),
            ('http://example.com',                  'http://'),
            ('http://example.com/index.html',       'http://'),
            ('localhost',                           'http://'),
            ('localhost/index.html',                'http://'),

            # some corner cases (default to http://)
            ('/',                                   'http://'),
            ('.../test',                            'http://'),

        ], start=1):
    t_method = create_guess_scheme_t(args)
    t_method.__name__ = 'test_uri_%03d' % k
    setattr (GuessSchemeTest, t_method.__name__, t_method)

# TODO: the following tests do not pass with current implementation
for k, args in enumerate ([
            ('C:\absolute\path\to\a\file.html',     'file://',
             'Windows filepath are not supported for scrapy shell'),
        ], start=1):
    t_method = create_skipped_scheme_t(args)
    t_method.__name__ = 'test_uri_skipped_%03d' % k
    setattr (GuessSchemeTest, t_method.__name__, t_method)


class M3(object):

    def process(self, response, request, spider):
        pass


class ChunkSize3PickleFifoDiskQueueTest(PickleFifoDiskQueueTest):
    chunksize = 3

class TestSpider(Spider):
    http_user = 'foo'
    http_pass = 'bar'


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

class TestCloseSpider(TestCase):

    def setUp(self):
        self.mockserver = MockServer()
        self.mockserver.__enter__()

    def tearDown(self):
        self.mockserver.__exit__(None, None, None)

    @defer.inlineCallbacks
    def test_closespider_itemcount(self):
        close_on = 5
        crawler = get_crawler(ItemSpider, {'CLOSESPIDER_ITEMCOUNT': close_on})
        yield crawler.crawl()
        reason = crawler.spider.meta['close_reason']
        self.assertEqual(reason, 'closespider_itemcount')
        itemcount = crawler.stats.get_value('item_scraped_count')
        self.assertTrue(itemcount >= close_on)

    @defer.inlineCallbacks
    def test_closespider_pagecount(self):
        close_on = 5
        crawler = get_crawler(FollowAllSpider, {'CLOSESPIDER_PAGECOUNT': close_on})
        yield crawler.crawl()
        reason = crawler.spider.meta['close_reason']
        self.assertEqual(reason, 'closespider_pagecount')
        pagecount = crawler.stats.get_value('response_received_count')
        self.assertTrue(pagecount >= close_on)

    @defer.inlineCallbacks
    def test_closespider_errorcount(self):
        close_on = 5
        crawler = get_crawler(ErrorSpider, {'CLOSESPIDER_ERRORCOUNT': close_on})
        yield crawler.crawl(total=1000000)
        reason = crawler.spider.meta['close_reason']
        self.assertEqual(reason, 'closespider_errorcount')
        key = 'spider_exceptions/{name}'\
                .format(name=crawler.spider.exception_cls.__name__)
        errorcount = crawler.stats.get_value(key)
        self.assertTrue(errorcount >= close_on)

    @defer.inlineCallbacks
    def test_closespider_timeout(self):
        close_on = 0.1
        crawler = get_crawler(FollowAllSpider, {'CLOSESPIDER_TIMEOUT': close_on})
        yield crawler.crawl(total=1000000)
        reason = crawler.spider.meta['close_reason']
        self.assertEqual(reason, 'closespider_timeout')
        stats = crawler.stats
        start = stats.get_value('start_time')
        stop = stats.get_value('finish_time')
        diff = stop - start
        total_seconds = diff.seconds + diff.microseconds
        self.assertTrue(total_seconds >= close_on)

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

class MiscCommandsTest(CommandTest):

    def test_list(self):
        self.assertEqual(0, self.call('list'))


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
class MediaPipelineAllowRedirectSettingsTestCase(unittest.TestCase):

    def _assert_request_no3xx(self, pipeline_class, settings):
        pipe = pipeline_class(settings=Settings(settings))
        request = Request('http://url')
        pipe._modify_media_request(request)

        self.assertIn('handle_httpstatus_list', request.meta)
        for status, check in [
                (200, True),

                # These are the status codes we want
                # the downloader to handle itself
                (301, False),
                (302, False),
                (302, False),
                (307, False),
                (308, False),

                # we still want to get 4xx and 5xx
                (400, True),
                (404, True),
                (500, True)]:
            if check:
                self.assertIn(status, request.meta['handle_httpstatus_list'])
            else:
                self.assertNotIn(status, request.meta['handle_httpstatus_list'])

    def test_standard_setting(self):
        self._assert_request_no3xx(
            MediaPipeline,
            {
                'MEDIA_ALLOW_REDIRECTS': True
            })

    def test_subclass_standard_setting(self):

        class UserDefinedPipeline(MediaPipeline):
            pass

        self._assert_request_no3xx(
            UserDefinedPipeline,
            {
                'MEDIA_ALLOW_REDIRECTS': True
            })

    def test_subclass_specific_setting(self):

        class UserDefinedPipeline(MediaPipeline):
            pass

        self._assert_request_no3xx(
            UserDefinedPipeline,
            {
                'USERDEFINEDPIPELINE_MEDIA_ALLOW_REDIRECTS': True
            })

class HttpDownloadHandlerMock(object):
    def __init__(self, settings):
        pass

    def download_request(self, request, spider):
        return request


class SendCatchLogDeferredTest2(SendCatchLogTest):

    def ok_handler(self, arg, handlers_called):
        handlers_called.add(self.ok_handler)
        assert arg == 'test'
        d = defer.Deferred()
        reactor.callLater(0, d.callback, "OK")
        return d

    def _get_result(self, signal, *a, **kw):
        return send_catch_log_deferred(signal, *a, **kw)

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

class CrawlerProcessTest(BaseCrawlerTest):
    def test_crawler_process_accepts_dict(self):
        runner = CrawlerProcess({'foo': 'bar'})
        self.assertEqual(runner.settings['foo'], 'bar')
        self.assertOptionIsDefault(runner.settings, 'RETRY_ENABLED')

    def test_crawler_process_accepts_None(self):
        runner = CrawlerProcess()
        self.assertOptionIsDefault(runner.settings, 'RETRY_ENABLED')

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


class DeprecatedClassesTest(unittest.TestCase):

    def test_deprecated_warnings(self):
        for cls in [ScrapyHTMLTranslator, ScrapyGenericTranslator, ScrapyXPathExpr]:
            with warnings.catch_warnings(record=True) as w:
                obj = cls()
                self.assertIn('%s is deprecated' % cls.__name__, str(w[-1].message),
                              'Missing deprecate warning for %s' % cls.__name__)



class DeprecatedFilesPipeline(FilesPipeline):
    def file_key(self, url):
        media_guid = hashlib.sha1(to_bytes(url)).hexdigest()
        media_ext = os.path.splitext(url)[1]
        return 'empty/%s%s' % (media_guid, media_ext)


class FileDownloadCrawlTestCase(TestCase):
    pipeline_class = 'scrapy.pipelines.files.FilesPipeline'
    store_setting_key = 'FILES_STORE'
    media_key = 'files'
    media_urls_key = 'file_urls'
    expected_checksums = set([
        '5547178b89448faf0015a13f904c936e',
        'c2281c83670e31d8aaab7cb642b824db',
        'ed3f6538dc15d4d9179dae57319edc5f'])

    def setUp(self):
        self.mockserver = MockServer()
        self.mockserver.__enter__()

        # prepare a directory for storing files
        self.tmpmediastore = self.mktemp()
        os.mkdir(self.tmpmediastore)
        self.settings = {
            'ITEM_PIPELINES': {self.pipeline_class: 1},
            self.store_setting_key: self.tmpmediastore,
        }
        self.runner = CrawlerRunner(self.settings)
        self.items = []

    def tearDown(self):
        shutil.rmtree(self.tmpmediastore)
        self.items = []
        self.mockserver.__exit__(None, None, None)

    def _on_item_scraped(self, item):
        self.items.append(item)

    def _create_crawler(self, spider_class, **kwargs):
        crawler = self.runner.create_crawler(spider_class, **kwargs)
        crawler.signals.connect(self._on_item_scraped, signals.item_scraped)
        return crawler

    def _assert_files_downloaded(self, items, logs):
        self.assertEqual(len(items), 1)
        self.assertIn(self.media_key, items[0])

        # check that logs show the expected number of successful file downloads
        file_dl_success = 'File (downloaded): Downloaded file from'
        self.assertEqual(logs.count(file_dl_success), 3)

        # check that the images/files checksums are what we know they should be
        if self.expected_checksums is not None:
            checksums = set(
                i['checksum']
                    for item in items
                        for i in item[self.media_key])
            self.assertEqual(checksums, self.expected_checksums)

        # check that the image files where actually written to the media store
        for item in items:
            for i in item[self.media_key]:
                self.assertTrue(
                    os.path.exists(
                        os.path.join(self.tmpmediastore, i['path'])))

    def _assert_files_download_failure(self, crawler, items, code, logs):

        # check that the item does NOT have the "images/files" field populated
        self.assertEqual(len(items), 1)
        self.assertIn(self.media_key, items[0])
        self.assertFalse(items[0][self.media_key])

        # check that there was 1 successful fetch and 3 other responses with non-200 code
        self.assertEqual(crawler.stats.get_value('downloader/request_method_count/GET'), 4)
        self.assertEqual(crawler.stats.get_value('downloader/response_count'), 4)
        self.assertEqual(crawler.stats.get_value('downloader/response_status_count/200'), 1)
        self.assertEqual(crawler.stats.get_value('downloader/response_status_count/%d' % code), 3)

        # check that logs do show the failure on the file downloads
        file_dl_failure = 'File (code: %d): Error downloading file from' % code
        self.assertEqual(logs.count(file_dl_failure), 3)

        # check that no files were written to the media store
        self.assertEqual(os.listdir(self.tmpmediastore), [])

    @defer.inlineCallbacks
    def test_download_media(self):
        crawler = self._create_crawler(MediaDownloadSpider)
        with LogCapture() as log:
            yield crawler.crawl("http://localhost:8998/files/images/",
                media_key=self.media_key,
                media_urls_key=self.media_urls_key)
        self._assert_files_downloaded(self.items, str(log))

    @defer.inlineCallbacks
    def test_download_media_wrong_urls(self):
        crawler = self._create_crawler(BrokenLinksMediaDownloadSpider)
        with LogCapture() as log:
            yield crawler.crawl("http://localhost:8998/files/images/",
                media_key=self.media_key,
                media_urls_key=self.media_urls_key)
        self._assert_files_download_failure(crawler, self.items, 404, str(log))

    @defer.inlineCallbacks
    def test_download_media_redirected_default_failure(self):
        crawler = self._create_crawler(RedirectedMediaDownloadSpider)
        with LogCapture() as log:
            yield crawler.crawl("http://localhost:8998/files/images/",
                media_key=self.media_key,
                media_urls_key=self.media_urls_key)
        self._assert_files_download_failure(crawler, self.items, 302, str(log))

    @defer.inlineCallbacks
    def test_download_media_redirected_allowed(self):
        settings = dict(self.settings)
        settings.update({'MEDIA_ALLOW_REDIRECTS': True})
        self.runner = CrawlerRunner(settings)

        crawler = self._create_crawler(RedirectedMediaDownloadSpider)
        with LogCapture() as log:
            yield crawler.crawl("http://localhost:8998/files/images/",
                media_key=self.media_key,
                media_urls_key=self.media_urls_key)
        self._assert_files_downloaded(self.items, str(log))
        self.assertEqual(crawler.stats.get_value('downloader/response_status_count/302'), 3)


class TestRequestMetaNoReferrer(MixinNoReferrer, TestRefererMiddleware):
    req_meta = {'referrer_policy': POLICY_NO_REFERRER}


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

