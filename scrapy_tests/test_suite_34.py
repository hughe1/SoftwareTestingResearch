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


class TestRequestMetaDefault(MixinDefault, TestRefererMiddleware):
    req_meta = {'referrer_policy': POLICY_SCRAPY_DEFAULT}


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


class FTPFeedStorageTest(unittest.TestCase):

    def test_store(self):
        uri = os.environ.get('FEEDTEST_FTP_URI')
        path = os.environ.get('FEEDTEST_FTP_PATH')
        if not (uri and path):
            raise unittest.SkipTest("No FTP server available for testing")
        st = FTPFeedStorage(uri)
        verifyObject(IFeedStorage, st)
        return self._assert_stores(st, path)

    @defer.inlineCallbacks
    def _assert_stores(self, storage, path):
        spider = scrapy.Spider("default")
        file = storage.open(spider)
        file.write(b"content")
        yield storage.store(file)
        self.assertTrue(os.path.exists(path))
        try:
            with open(path, 'rb') as fp:
                self.assertEqual(fp.read(), b"content")
            # again, to check s3 objects are overwritten
            yield storage.store(BytesIO(b"new content"))
            with open(path, 'rb') as fp:
                self.assertEqual(fp.read(), b"new content")
        finally:
            os.unlink(path)


class HttpDownloadHandlerMock(object):
    def __init__(self, settings):
        pass

    def download_request(self, request, spider):
        return request


class DeprecatedHttpProxyTestCase(unittest.TestCase):
    """Old deprecated reference to http10 downloader handler"""
    download_handler_cls = HttpDownloadHandler


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


class Base:
    class LinkExtractorTestCase(unittest.TestCase):
        extractor_cls = None
        escapes_whitespace = False

        def setUp(self):
            body = get_testdata('link_extractor', 'sgml_linkextractor.html')
            self.response = HtmlResponse(url='http://example.com/index', body=body)

        def test_urls_type(self):
            ''' Test that the resulting urls are str objects '''
            lx = self.extractor_cls()
            self.assertTrue(all(isinstance(link.url, str)
                                for link in lx.extract_links(self.response)))

        def test_extract_all_links(self):
            lx = self.extractor_cls()
            if self.escapes_whitespace:
                page4_url = 'http://example.com/page%204.html'
            else:
                page4_url = 'http://example.com/page 4.html'

            self.assertEqual([link for link in lx.extract_links(self.response)], [
                Link(url='http://example.com/sample1.html', text=u''),
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
                Link(url='http://example.com/sample3.html', text=u'sample 3 text'),
                Link(url='http://example.com/sample3.html#foo', text='sample 3 repetition with fragment'),
                Link(url='http://www.google.com/something', text=u''),
                Link(url='http://example.com/innertag.html', text=u'inner tag'),
                Link(url=page4_url, text=u'href with whitespaces'),
            ])

        def test_extract_filter_allow(self):
            lx = self.extractor_cls(allow=('sample', ))
            self.assertEqual([link for link in lx.extract_links(self.response)], [
                Link(url='http://example.com/sample1.html', text=u''),
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
                Link(url='http://example.com/sample3.html', text=u'sample 3 text'),
                Link(url='http://example.com/sample3.html#foo', text='sample 3 repetition with fragment')
            ])

        def test_extract_filter_allow_with_duplicates(self):
            lx = self.extractor_cls(allow=('sample', ), unique=False)
            self.assertEqual([link for link in lx.extract_links(self.response)], [
                Link(url='http://example.com/sample1.html', text=u''),
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
                Link(url='http://example.com/sample3.html', text=u'sample 3 text'),
                Link(url='http://example.com/sample3.html', text=u'sample 3 repetition'),
                Link(url='http://example.com/sample3.html#foo', text='sample 3 repetition with fragment')
            ])

        def test_extract_filter_allow_with_duplicates_canonicalize(self):
            lx = self.extractor_cls(allow=('sample', ), unique=False,
                                    canonicalize=True)
            self.assertEqual([link for link in lx.extract_links(self.response)], [
                Link(url='http://example.com/sample1.html', text=u''),
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
                Link(url='http://example.com/sample3.html', text=u'sample 3 text'),
                Link(url='http://example.com/sample3.html', text=u'sample 3 repetition'),
                Link(url='http://example.com/sample3.html', text='sample 3 repetition with fragment')
            ])

        def test_extract_filter_allow_no_duplicates_canonicalize(self):
            lx = self.extractor_cls(allow=('sample',), unique=True,
                                    canonicalize=True)
            self.assertEqual([link for link in lx.extract_links(self.response)], [
                Link(url='http://example.com/sample1.html', text=u''),
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
                Link(url='http://example.com/sample3.html', text=u'sample 3 text'),
            ])

        def test_extract_filter_allow_and_deny(self):
            lx = self.extractor_cls(allow=('sample', ), deny=('3', ))
            self.assertEqual([link for link in lx.extract_links(self.response)], [
                Link(url='http://example.com/sample1.html', text=u''),
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
            ])

        def test_extract_filter_allowed_domains(self):
            lx = self.extractor_cls(allow_domains=('google.com', ))
            self.assertEqual([link for link in lx.extract_links(self.response)], [
                Link(url='http://www.google.com/something', text=u''),
            ])

        def test_extraction_using_single_values(self):
            '''Test the extractor's behaviour among different situations'''

            lx = self.extractor_cls(allow='sample')
            self.assertEqual([link for link in lx.extract_links(self.response)], [
                Link(url='http://example.com/sample1.html', text=u''),
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
                Link(url='http://example.com/sample3.html', text=u'sample 3 text'),
                Link(url='http://example.com/sample3.html#foo',
                     text='sample 3 repetition with fragment')
            ])

            lx = self.extractor_cls(allow='sample', deny='3')
            self.assertEqual([link for link in lx.extract_links(self.response)], [
                Link(url='http://example.com/sample1.html', text=u''),
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
            ])

            lx = self.extractor_cls(allow_domains='google.com')
            self.assertEqual([link for link in lx.extract_links(self.response)], [
                Link(url='http://www.google.com/something', text=u''),
            ])

            lx = self.extractor_cls(deny_domains='example.com')
            self.assertEqual([link for link in lx.extract_links(self.response)], [
                Link(url='http://www.google.com/something', text=u''),
            ])

        def test_nofollow(self):
            '''Test the extractor's behaviour for links with rel="nofollow"'''

            html = b"""<html><head><title>Page title<title>
            <body>
            <div class='links'>
            <p><a href="/about.html">About us</a></p>
            </div>
            <div>
            <p><a href="/follow.html">Follow this link</a></p>
            </div>
            <div>
            <p><a href="/nofollow.html" rel="nofollow">Dont follow this one</a></p>
            </div>
            <div>
            <p><a href="/nofollow2.html" rel="blah">Choose to follow or not</a></p>
            </div>
            <div>
            <p><a href="http://google.com/something" rel="external nofollow">External link not to follow</a></p>
            </div>
            </body></html>"""
            response = HtmlResponse("http://example.org/somepage/index.html", body=html)

            lx = self.extractor_cls()
            self.assertEqual(lx.extract_links(response), [
                Link(url='http://example.org/about.html', text=u'About us'),
                Link(url='http://example.org/follow.html', text=u'Follow this link'),
                Link(url='http://example.org/nofollow.html', text=u'Dont follow this one', nofollow=True),
                Link(url='http://example.org/nofollow2.html', text=u'Choose to follow or not'),
                Link(url='http://google.com/something', text=u'External link not to follow', nofollow=True),
            ])

        def test_matches(self):
            url1 = 'http://lotsofstuff.com/stuff1/index'
            url2 = 'http://evenmorestuff.com/uglystuff/index'

            lx = self.extractor_cls(allow=(r'stuff1', ))
            self.assertEqual(lx.matches(url1), True)
            self.assertEqual(lx.matches(url2), False)

            lx = self.extractor_cls(deny=(r'uglystuff', ))
            self.assertEqual(lx.matches(url1), True)
            self.assertEqual(lx.matches(url2), False)

            lx = self.extractor_cls(allow_domains=('evenmorestuff.com', ))
            self.assertEqual(lx.matches(url1), False)
            self.assertEqual(lx.matches(url2), True)

            lx = self.extractor_cls(deny_domains=('lotsofstuff.com', ))
            self.assertEqual(lx.matches(url1), False)
            self.assertEqual(lx.matches(url2), True)

            lx = self.extractor_cls(allow=('blah1',), deny=('blah2',),
                                   allow_domains=('blah1.com',),
                                   deny_domains=('blah2.com',))
            self.assertEqual(lx.matches('http://blah1.com/blah1'), True)
            self.assertEqual(lx.matches('http://blah1.com/blah2'), False)
            self.assertEqual(lx.matches('http://blah2.com/blah1'), False)
            self.assertEqual(lx.matches('http://blah2.com/blah2'), False)

        def test_restrict_xpaths(self):
            lx = self.extractor_cls(restrict_xpaths=('//div[@id="subwrapper"]', ))
            self.assertEqual([link for link in lx.extract_links(self.response)], [
                Link(url='http://example.com/sample1.html', text=u''),
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
            ])

        def test_restrict_xpaths_encoding(self):
            """Test restrict_xpaths with encodings"""
            html = b"""<html><head><title>Page title<title>
            <body><p><a href="item/12.html">Item 12</a></p>
            <div class='links'>
            <p><a href="/about.html">About us\xa3</a></p>
            </div>
            <div>
            <p><a href="/nofollow.html">This shouldn't be followed</a></p>
            </div>
            </body></html>"""
            response = HtmlResponse("http://example.org/somepage/index.html", body=html, encoding='windows-1252')

            lx = self.extractor_cls(restrict_xpaths="//div[@class='links']")
            self.assertEqual(lx.extract_links(response),
                             [Link(url='http://example.org/about.html', text=u'About us\xa3')])

        def test_restrict_xpaths_with_html_entities(self):
            html = b'<html><body><p><a href="/&hearts;/you?c=&euro;">text</a></p></body></html>'
            response = HtmlResponse("http://example.org/somepage/index.html", body=html, encoding='iso8859-15')
            links = self.extractor_cls(restrict_xpaths='//p').extract_links(response)
            self.assertEqual(links,
                             [Link(url='http://example.org/%E2%99%A5/you?c=%E2%82%AC', text=u'text')])

        def test_restrict_xpaths_concat_in_handle_data(self):
            """html entities cause SGMLParser to call handle_data hook twice"""
            body = b"""<html><body><div><a href="/foo">&gt;\xbe\xa9&lt;\xb6\xab</a></body></html>"""
            response = HtmlResponse("http://example.org", body=body, encoding='gb18030')
            lx = self.extractor_cls(restrict_xpaths="//div")
            self.assertEqual(lx.extract_links(response),
                             [Link(url='http://example.org/foo', text=u'>\u4eac<\u4e1c',
                                   fragment='', nofollow=False)])

        def test_restrict_css(self):
            lx = self.extractor_cls(restrict_css=('#subwrapper a',))
            self.assertEqual(lx.extract_links(self.response), [
                Link(url='http://example.com/sample2.html', text=u'sample 2')
            ])

        def test_restrict_css_and_restrict_xpaths_together(self):
            lx = self.extractor_cls(restrict_xpaths=('//div[@id="subwrapper"]', ),
                                    restrict_css=('#subwrapper + a', ))
            self.assertEqual([link for link in lx.extract_links(self.response)], [
                Link(url='http://example.com/sample1.html', text=u''),
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
                Link(url='http://example.com/sample3.html', text=u'sample 3 text'),
            ])

        def test_area_tag_with_unicode_present(self):
            body = b"""<html><body>\xbe\xa9<map><area href="http://example.org/foo" /></map></body></html>"""
            response = HtmlResponse("http://example.org", body=body, encoding='utf-8')
            lx = self.extractor_cls()
            lx.extract_links(response)
            lx.extract_links(response)
            lx.extract_links(response)
            self.assertEqual(lx.extract_links(response),
                             [Link(url='http://example.org/foo', text=u'',
                                   fragment='', nofollow=False)])

        def test_encoded_url(self):
            body = b"""<html><body><div><a href="?page=2">BinB</a></body></html>"""
            response = HtmlResponse("http://known.fm/AC%2FDC/", body=body, encoding='utf8')
            lx = self.extractor_cls()
            self.assertEqual(lx.extract_links(response), [
                Link(url='http://known.fm/AC%2FDC/?page=2', text=u'BinB', fragment='', nofollow=False),
            ])

        def test_encoded_url_in_restricted_xpath(self):
            body = b"""<html><body><div><a href="?page=2">BinB</a></body></html>"""
            response = HtmlResponse("http://known.fm/AC%2FDC/", body=body, encoding='utf8')
            lx = self.extractor_cls(restrict_xpaths="//div")
            self.assertEqual(lx.extract_links(response), [
                Link(url='http://known.fm/AC%2FDC/?page=2', text=u'BinB', fragment='', nofollow=False),
            ])

        def test_ignored_extensions(self):
            # jpg is ignored by default
            html = b"""<a href="page.html">asd</a> and <a href="photo.jpg">"""
            response = HtmlResponse("http://example.org/", body=html)
            lx = self.extractor_cls()
            self.assertEqual(lx.extract_links(response), [
                Link(url='http://example.org/page.html', text=u'asd'),
            ])

            # override denied extensions
            lx = self.extractor_cls(deny_extensions=['html'])
            self.assertEqual(lx.extract_links(response), [
                Link(url='http://example.org/photo.jpg'),
            ])

        def test_process_value(self):
            """Test restrict_xpaths with encodings"""
            html = b"""
            <a href="javascript:goToPage('../other/page.html','photo','width=600,height=540,scrollbars'); return false">Link text</a>
            <a href="/about.html">About us</a>
            """
            response = HtmlResponse("http://example.org/somepage/index.html", body=html, encoding='windows-1252')

            def process_value(value):
                m = re.search("javascript:goToPage\('(.*?)'", value)
                if m:
                    return m.group(1)

            lx = self.extractor_cls(process_value=process_value)
            self.assertEqual(lx.extract_links(response),
                             [Link(url='http://example.org/other/page.html', text='Link text')])

        def test_base_url_with_restrict_xpaths(self):
            html = b"""<html><head><title>Page title<title><base href="http://otherdomain.com/base/" />
            <body><p><a href="item/12.html">Item 12</a></p>
            </body></html>"""
            response = HtmlResponse("http://example.org/somepage/index.html", body=html)
            lx = self.extractor_cls(restrict_xpaths="//p")
            self.assertEqual(lx.extract_links(response),
                             [Link(url='http://otherdomain.com/base/item/12.html', text='Item 12')])

        def test_attrs(self):
            lx = self.extractor_cls(attrs="href")
            if self.escapes_whitespace:
                page4_url = 'http://example.com/page%204.html'
            else:
                page4_url = 'http://example.com/page 4.html'

            self.assertEqual(lx.extract_links(self.response), [
                Link(url='http://example.com/sample1.html', text=u''),
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
                Link(url='http://example.com/sample3.html', text=u'sample 3 text'),
                Link(url='http://example.com/sample3.html#foo', text='sample 3 repetition with fragment'),
                Link(url='http://www.google.com/something', text=u''),
                Link(url='http://example.com/innertag.html', text=u'inner tag'),
                Link(url=page4_url, text=u'href with whitespaces'),
            ])

            lx = self.extractor_cls(attrs=("href","src"), tags=("a","area","img"), deny_extensions=())
            self.assertEqual(lx.extract_links(self.response), [
                Link(url='http://example.com/sample1.html', text=u''),
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
                Link(url='http://example.com/sample2.jpg', text=u''),
                Link(url='http://example.com/sample3.html', text=u'sample 3 text'),
                Link(url='http://example.com/sample3.html#foo', text='sample 3 repetition with fragment'),
                Link(url='http://www.google.com/something', text=u''),
                Link(url='http://example.com/innertag.html', text=u'inner tag'),
                Link(url=page4_url, text=u'href with whitespaces'),
            ])

            lx = self.extractor_cls(attrs=None)
            self.assertEqual(lx.extract_links(self.response), [])

        def test_tags(self):
            html = b"""<html><area href="sample1.html"></area><a href="sample2.html">sample 2</a><img src="sample2.jpg"/></html>"""
            response = HtmlResponse("http://example.com/index.html", body=html)

            lx = self.extractor_cls(tags=None)
            self.assertEqual(lx.extract_links(response), [])

            lx = self.extractor_cls()
            self.assertEqual(lx.extract_links(response), [
                Link(url='http://example.com/sample1.html', text=u''),
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
            ])

            lx = self.extractor_cls(tags="area")
            self.assertEqual(lx.extract_links(response), [
                Link(url='http://example.com/sample1.html', text=u''),
            ])

            lx = self.extractor_cls(tags="a")
            self.assertEqual(lx.extract_links(response), [
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
            ])

            lx = self.extractor_cls(tags=("a","img"), attrs=("href", "src"), deny_extensions=())
            self.assertEqual(lx.extract_links(response), [
                Link(url='http://example.com/sample2.html', text=u'sample 2'),
                Link(url='http://example.com/sample2.jpg', text=u''),
            ])

        def test_tags_attrs(self):
            html = b"""
            <html><body>
            <div id="item1" data-url="get?id=1"><a href="#">Item 1</a></div>
            <div id="item2" data-url="get?id=2"><a href="#">Item 2</a></div>
            </body></html>
            """
            response = HtmlResponse("http://example.com/index.html", body=html)

            lx = self.extractor_cls(tags='div', attrs='data-url')
            self.assertEqual(lx.extract_links(response), [
                Link(url='http://example.com/get?id=1', text=u'Item 1', fragment='', nofollow=False),
                Link(url='http://example.com/get?id=2', text=u'Item 2', fragment='', nofollow=False)
            ])

            lx = self.extractor_cls(tags=('div',), attrs=('data-url',))
            self.assertEqual(lx.extract_links(response), [
                Link(url='http://example.com/get?id=1', text=u'Item 1', fragment='', nofollow=False),
                Link(url='http://example.com/get?id=2', text=u'Item 2', fragment='', nofollow=False)
            ])

        def test_xhtml(self):
            xhtml = b"""
    <?xml version="1.0"?>
    <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"
        "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
    <html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
    <head>
        <title>XHTML document title</title>
    </head>
    <body>
        <div class='links'>
        <p><a href="/about.html">About us</a></p>
        </div>
        <div>
        <p><a href="/follow.html">Follow this link</a></p>
        </div>
        <div>
        <p><a href="/nofollow.html" rel="nofollow">Dont follow this one</a></p>
        </div>
        <div>
        <p><a href="/nofollow2.html" rel="blah">Choose to follow or not</a></p>
        </div>
        <div>
        <p><a href="http://google.com/something" rel="external nofollow">External link not to follow</a></p>
        </div>
    </body>
    </html>
            """

            response = HtmlResponse("http://example.com/index.xhtml", body=xhtml)

            lx = self.extractor_cls()
            self.assertEqual(lx.extract_links(response),
                             [Link(url='http://example.com/about.html', text=u'About us', fragment='', nofollow=False),
                              Link(url='http://example.com/follow.html', text=u'Follow this link', fragment='', nofollow=False),
                              Link(url='http://example.com/nofollow.html', text=u'Dont follow this one', fragment='', nofollow=True),
                              Link(url='http://example.com/nofollow2.html', text=u'Choose to follow or not', fragment='', nofollow=False),
                              Link(url='http://google.com/something', text=u'External link not to follow', nofollow=True)]
                            )

            response = XmlResponse("http://example.com/index.xhtml", body=xhtml)

            lx = self.extractor_cls()
            self.assertEqual(lx.extract_links(response),
                             [Link(url='http://example.com/about.html', text=u'About us', fragment='', nofollow=False),
                              Link(url='http://example.com/follow.html', text=u'Follow this link', fragment='', nofollow=False),
                              Link(url='http://example.com/nofollow.html', text=u'Dont follow this one', fragment='', nofollow=True),
                              Link(url='http://example.com/nofollow2.html', text=u'Choose to follow or not', fragment='', nofollow=False),
                              Link(url='http://google.com/something', text=u'External link not to follow', nofollow=True)]
                            )

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


class FilesPipelineTestCaseCustomSettings(unittest.TestCase):
    default_cls_settings = {
        "EXPIRES": 90,
        "FILES_URLS_FIELD": "file_urls",
        "FILES_RESULT_FIELD": "files"
    }
    file_cls_attr_settings_map = {
        ("EXPIRES", "FILES_EXPIRES", "expires"),
        ("FILES_URLS_FIELD", "FILES_URLS_FIELD", "files_urls_field"),
        ("FILES_RESULT_FIELD", "FILES_RESULT_FIELD", "files_result_field")
    }

    def setUp(self):
        self.tempdir = mkdtemp()

    def tearDown(self):
        rmtree(self.tempdir)

    def _generate_fake_settings(self, prefix=None):

        def random_string():
            return "".join([chr(random.randint(97, 123)) for _ in range(10)])

        settings = {
            "FILES_EXPIRES": random.randint(100, 1000),
            "FILES_URLS_FIELD": random_string(),
            "FILES_RESULT_FIELD": random_string(),
            "FILES_STORE": self.tempdir
        }
        if not prefix:
            return settings

        return {prefix.upper() + "_" + k if k != "FILES_STORE" else k: v for k, v in settings.items()}

    def _generate_fake_pipeline(self):

        class UserDefinedFilePipeline(FilesPipeline):
            EXPIRES = 1001
            FILES_URLS_FIELD = "alfa"
            FILES_RESULT_FIELD = "beta"

        return UserDefinedFilePipeline

    def test_different_settings_for_different_instances(self):
        """
        If there are different instances with different settings they should keep
        different settings.
        """
        custom_settings = self._generate_fake_settings()
        another_pipeline = FilesPipeline.from_settings(Settings(custom_settings))
        one_pipeline = FilesPipeline(self.tempdir)
        for pipe_attr, settings_attr, pipe_ins_attr in self.file_cls_attr_settings_map:
            default_value = self.default_cls_settings[pipe_attr]
            self.assertEqual(getattr(one_pipeline, pipe_attr), default_value)
            custom_value = custom_settings[settings_attr]
            self.assertNotEqual(default_value, custom_value)
            self.assertEqual(getattr(another_pipeline, pipe_ins_attr), custom_value)

    def test_subclass_attributes_preserved_if_no_settings(self):
        """
        If subclasses override class attributes and there are no special settings those values should be kept.
        """
        pipe_cls = self._generate_fake_pipeline()
        pipe = pipe_cls.from_settings(Settings({"FILES_STORE": self.tempdir}))
        for pipe_attr, settings_attr, pipe_ins_attr in self.file_cls_attr_settings_map:
            custom_value = getattr(pipe, pipe_ins_attr)
            self.assertNotEqual(custom_value, self.default_cls_settings[pipe_attr])
            self.assertEqual(getattr(pipe, pipe_ins_attr), getattr(pipe, pipe_attr))

    def test_subclass_attrs_preserved_custom_settings(self):
        """
        If file settings are defined but they are not defined for subclass
        settings should be preserved.
        """
        pipeline_cls = self._generate_fake_pipeline()
        settings = self._generate_fake_settings()
        pipeline = pipeline_cls.from_settings(Settings(settings))
        for pipe_attr, settings_attr, pipe_ins_attr in self.file_cls_attr_settings_map:
            value = getattr(pipeline, pipe_ins_attr)
            setting_value = settings.get(settings_attr)
            self.assertNotEqual(value, self.default_cls_settings[pipe_attr])
            self.assertEqual(value, setting_value)

    def test_no_custom_settings_for_subclasses(self):
        """
        If there are no settings for subclass and no subclass attributes, pipeline should use
        attributes of base class.
        """
        class UserDefinedFilesPipeline(FilesPipeline):
            pass

        user_pipeline = UserDefinedFilesPipeline.from_settings(Settings({"FILES_STORE": self.tempdir}))
        for pipe_attr, settings_attr, pipe_ins_attr in self.file_cls_attr_settings_map:
            # Values from settings for custom pipeline should be set on pipeline instance.
            custom_value = self.default_cls_settings.get(pipe_attr.upper())
            self.assertEqual(getattr(user_pipeline, pipe_ins_attr), custom_value)

    def test_custom_settings_for_subclasses(self):
        """
        If there are custom settings for subclass and NO class attributes, pipeline should use custom
        settings.
        """
        class UserDefinedFilesPipeline(FilesPipeline):
            pass

        prefix = UserDefinedFilesPipeline.__name__.upper()
        settings = self._generate_fake_settings(prefix=prefix)
        user_pipeline = UserDefinedFilesPipeline.from_settings(Settings(settings))
        for pipe_attr, settings_attr, pipe_inst_attr in self.file_cls_attr_settings_map:
            # Values from settings for custom pipeline should be set on pipeline instance.
            custom_value = settings.get(prefix + "_" + settings_attr)
            self.assertNotEqual(custom_value, self.default_cls_settings[pipe_attr])
            self.assertEqual(getattr(user_pipeline, pipe_inst_attr), custom_value)

    def test_custom_settings_and_class_attrs_for_subclasses(self):
        """
        If there are custom settings for subclass AND class attributes
        setting keys are preferred and override attributes.
        """
        pipeline_cls = self._generate_fake_pipeline()
        prefix = pipeline_cls.__name__.upper()
        settings = self._generate_fake_settings(prefix=prefix)
        user_pipeline = pipeline_cls.from_settings(Settings(settings))
        for pipe_cls_attr, settings_attr, pipe_inst_attr  in self.file_cls_attr_settings_map:
            custom_value = settings.get(prefix + "_" + settings_attr)
            self.assertNotEqual(custom_value, self.default_cls_settings[pipe_cls_attr])
            self.assertEqual(getattr(user_pipeline, pipe_inst_attr), custom_value)

    def test_cls_attrs_with_DEFAULT_prefix(self):
        class UserDefinedFilesPipeline(FilesPipeline):
            DEFAULT_FILES_RESULT_FIELD = "this"
            DEFAULT_FILES_URLS_FIELD = "that"

        pipeline = UserDefinedFilesPipeline.from_settings(Settings({"FILES_STORE": self.tempdir}))
        self.assertEqual(pipeline.files_result_field, "this")
        self.assertEqual(pipeline.files_urls_field, "that")


    def test_user_defined_subclass_default_key_names(self):
        """Test situation when user defines subclass of FilesPipeline,
        but uses attribute names for default pipeline (without prefixing
        them with pipeline class name).
        """
        settings = self._generate_fake_settings()

        class UserPipe(FilesPipeline):
            pass

        pipeline_cls = UserPipe.from_settings(Settings(settings))

        for pipe_attr, settings_attr, pipe_inst_attr in self.file_cls_attr_settings_map:
            expected_value = settings.get(settings_attr)
            self.assertEqual(getattr(pipeline_cls, pipe_inst_attr),
                             expected_value)


class RedirectMiddlewareTest(unittest.TestCase):

    def setUp(self):
        self.crawler = get_crawler(Spider)
        self.spider = self.crawler._create_spider('foo')
        self.mw = RedirectMiddleware.from_crawler(self.crawler)

    def test_priority_adjust(self):
        req = Request('http://a.com')
        rsp = Response('http://a.com', headers={'Location': 'http://a.com/redirected'}, status=301)
        req2 = self.mw.process_response(req, rsp, self.spider)
        assert req2.priority > req.priority

    def test_redirect_3xx_permanent(self):
        def _test(method, status=301):
            url = 'http://www.example.com/{}'.format(status)
            url2 = 'http://www.example.com/redirected'
            req = Request(url, method=method)
            rsp = Response(url, headers={'Location': url2}, status=status)

            req2 = self.mw.process_response(req, rsp, self.spider)
            assert isinstance(req2, Request)
            self.assertEqual(req2.url, url2)
            self.assertEqual(req2.method, method)

            # response without Location header but with status code is 3XX should be ignored
            del rsp.headers['Location']
            assert self.mw.process_response(req, rsp, self.spider) is rsp

        _test('GET')
        _test('POST')
        _test('HEAD')

        _test('GET', status=307)
        _test('POST', status=307)
        _test('HEAD', status=307)

        _test('GET', status=308)
        _test('POST', status=308)
        _test('HEAD', status=308)

    def test_dont_redirect(self):
        url = 'http://www.example.com/301'
        url2 = 'http://www.example.com/redirected'
        req = Request(url, meta={'dont_redirect': True})
        rsp = Response(url, headers={'Location': url2}, status=301)

        r = self.mw.process_response(req, rsp, self.spider)
        assert isinstance(r, Response)
        assert r is rsp

        # Test that it redirects when dont_redirect is False
        req = Request(url, meta={'dont_redirect': False})
        rsp = Response(url2, status=200)

        r = self.mw.process_response(req, rsp, self.spider)
        assert isinstance(r, Response)
        assert r is rsp


    def test_redirect_302(self):
        url = 'http://www.example.com/302'
        url2 = 'http://www.example.com/redirected2'
        req = Request(url, method='POST', body='test',
            headers={'Content-Type': 'text/plain', 'Content-length': '4'})
        rsp = Response(url, headers={'Location': url2}, status=302)

        req2 = self.mw.process_response(req, rsp, self.spider)
        assert isinstance(req2, Request)
        self.assertEqual(req2.url, url2)
        self.assertEqual(req2.method, 'GET')
        assert 'Content-Type' not in req2.headers, \
            "Content-Type header must not be present in redirected request"
        assert 'Content-Length' not in req2.headers, \
            "Content-Length header must not be present in redirected request"
        assert not req2.body, \
            "Redirected body must be empty, not '%s'" % req2.body

        # response without Location header but with status code is 3XX should be ignored
        del rsp.headers['Location']
        assert self.mw.process_response(req, rsp, self.spider) is rsp

    def test_redirect_302_head(self):
        url = 'http://www.example.com/302'
        url2 = 'http://www.example.com/redirected2'
        req = Request(url, method='HEAD')
        rsp = Response(url, headers={'Location': url2}, status=302)

        req2 = self.mw.process_response(req, rsp, self.spider)
        assert isinstance(req2, Request)
        self.assertEqual(req2.url, url2)
        self.assertEqual(req2.method, 'HEAD')

        # response without Location header but with status code is 3XX should be ignored
        del rsp.headers['Location']
        assert self.mw.process_response(req, rsp, self.spider) is rsp


    def test_max_redirect_times(self):
        self.mw.max_redirect_times = 1
        req = Request('http://scrapytest.org/302')
        rsp = Response('http://scrapytest.org/302', headers={'Location': '/redirected'}, status=302)

        req = self.mw.process_response(req, rsp, self.spider)
        assert isinstance(req, Request)
        assert 'redirect_times' in req.meta
        self.assertEqual(req.meta['redirect_times'], 1)
        self.assertRaises(IgnoreRequest, self.mw.process_response, req, rsp, self.spider)

    def test_ttl(self):
        self.mw.max_redirect_times = 100
        req = Request('http://scrapytest.org/302', meta={'redirect_ttl': 1})
        rsp = Response('http://www.scrapytest.org/302', headers={'Location': '/redirected'}, status=302)

        req = self.mw.process_response(req, rsp, self.spider)
        assert isinstance(req, Request)
        self.assertRaises(IgnoreRequest, self.mw.process_response, req, rsp, self.spider)

    def test_redirect_urls(self):
        req1 = Request('http://scrapytest.org/first')
        rsp1 = Response('http://scrapytest.org/first', headers={'Location': '/redirected'}, status=302)
        req2 = self.mw.process_response(req1, rsp1, self.spider)
        rsp2 = Response('http://scrapytest.org/redirected', headers={'Location': '/redirected2'}, status=302)
        req3 = self.mw.process_response(req2, rsp2, self.spider)

        self.assertEqual(req2.url, 'http://scrapytest.org/redirected')
        self.assertEqual(req2.meta['redirect_urls'], ['http://scrapytest.org/first'])
        self.assertEqual(req3.url, 'http://scrapytest.org/redirected2')
        self.assertEqual(req3.meta['redirect_urls'], ['http://scrapytest.org/first', 'http://scrapytest.org/redirected'])

    def test_spider_handling(self):
        smartspider = self.crawler._create_spider('smarty')
        smartspider.handle_httpstatus_list = [404, 301, 302]
        url = 'http://www.example.com/301'
        url2 = 'http://www.example.com/redirected'
        req = Request(url)
        rsp = Response(url, headers={'Location': url2}, status=301)
        r = self.mw.process_response(req, rsp, smartspider)
        self.assertIs(r, rsp)

    def test_request_meta_handling(self):
        url = 'http://www.example.com/301'
        url2 = 'http://www.example.com/redirected'
        def _test_passthrough(req):
            rsp = Response(url, headers={'Location': url2}, status=301, request=req)
            r = self.mw.process_response(req, rsp, self.spider)
            self.assertIs(r, rsp)
        _test_passthrough(Request(url, meta={'handle_httpstatus_list':
                                                           [404, 301, 302]}))
        _test_passthrough(Request(url, meta={'handle_httpstatus_all': True}))

    def test_latin1_location(self):
        req = Request('http://scrapytest.org/first')
        latin1_location = u'/ação'.encode('latin1')  # HTTP historically supports latin1
        resp = Response('http://scrapytest.org/first', headers={'Location': latin1_location}, status=302)
        req_result = self.mw.process_response(req, resp, self.spider)
        perc_encoded_utf8_url = 'http://scrapytest.org/a%E7%E3o'
        self.assertEqual(perc_encoded_utf8_url, req_result.url)

    def test_utf8_location(self):
        req = Request('http://scrapytest.org/first')
        utf8_location = u'/ação'.encode('utf-8')  # header using UTF-8 encoding
        resp = Response('http://scrapytest.org/first', headers={'Location': utf8_location}, status=302)
        req_result = self.mw.process_response(req, resp, self.spider)
        perc_encoded_utf8_url = 'http://scrapytest.org/a%C3%A7%C3%A3o'
        self.assertEqual(perc_encoded_utf8_url, req_result.url)


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

class InitSpiderTest(SpiderTest):

    spider_class = InitSpider


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


class LoggingContribTest(unittest.TestCase):

    def setUp(self):
        self.formatter = LogFormatter()
        self.spider = Spider('default')

    def test_crawled(self):
        req = Request("http://www.example.com")
        res = Response("http://www.example.com")
        logkws = self.formatter.crawled(req, res, self.spider)
        logline = logkws['msg'] % logkws['args']
        self.assertEqual(logline,
            "Crawled (200) <GET http://www.example.com> (referer: None)")

        req = Request("http://www.example.com", headers={'referer': 'http://example.com'})
        res = Response("http://www.example.com", flags=['cached'])
        logkws = self.formatter.crawled(req, res, self.spider)
        logline = logkws['msg'] % logkws['args']
        self.assertEqual(logline,
            "Crawled (200) <GET http://www.example.com> (referer: http://example.com) ['cached']")

    def test_flags_in_request(self):
        req = Request("http://www.example.com", flags=['test','flag'])
        res = Response("http://www.example.com")
        logkws = self.formatter.crawled(req, res, self.spider)
        logline = logkws['msg'] % logkws['args']
        self.assertEqual(logline,
        "Crawled (200) <GET http://www.example.com> ['test', 'flag'] (referer: None)")

    def test_dropped(self):
        item = {}
        exception = Exception(u"\u2018")
        response = Response("http://www.example.com")
        logkws = self.formatter.dropped(item, exception, response, self.spider)
        logline = logkws['msg'] % logkws['args']
        lines = logline.splitlines()
        assert all(isinstance(x, six.text_type) for x in lines)
        self.assertEqual(lines, [u"Dropped: \u2018", '{}'])

    def test_scraped(self):
        item = CustomItem()
        item['name'] = u'\xa3'
        response = Response("http://www.example.com")
        logkws = self.formatter.scraped(item, response, self.spider)
        logline = logkws['msg'] % logkws['args']
        lines = logline.splitlines()
        assert all(isinstance(x, six.text_type) for x in lines)
        self.assertEqual(lines, [u"Scraped from <200 http://www.example.com>", u'name: \xa3'])


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


class DbmStorageTest(DefaultStorageTest):

    storage_class = 'scrapy.extensions.httpcache.DbmCacheStorage'


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



class EncodingResource(resource.Resource):
    out_encoding = 'cp1251'

    def render(self, request):
        body = to_unicode(request.content.read())
        request.setHeader(b'content-encoding', self.out_encoding)
        return body.encode(self.out_encoding)


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

class MiddlewareManagerTest(unittest.TestCase):

    def test_init(self):
        m1, m2, m3 = M1(), M2(), M3()
        mwman = TestMiddlewareManager(m1, m2, m3)
        self.assertEqual(mwman.methods['open_spider'], [m1.open_spider, m2.open_spider])
        self.assertEqual(mwman.methods['close_spider'], [m2.close_spider, m1.close_spider])
        self.assertEqual(mwman.methods['process'], [m1.process, m3.process])

    def test_methods(self):
        mwman = TestMiddlewareManager(M1(), M2(), M3())
        if six.PY2:
            self.assertEqual([x.im_class for x in mwman.methods['open_spider']],
                [M1, M2])
            self.assertEqual([x.im_class for x in mwman.methods['close_spider']],
                [M2, M1])
            self.assertEqual([x.im_class for x in mwman.methods['process']],
                [M1, M3])
        else:
            self.assertEqual([x.__self__.__class__ for x in mwman.methods['open_spider']],
                [M1, M2])
            self.assertEqual([x.__self__.__class__ for x in mwman.methods['close_spider']],
                [M2, M1])
            self.assertEqual([x.__self__.__class__ for x in mwman.methods['process']],
                [M1, M3])

    def test_enabled(self):
        m1, m2, m3 = M1(), M2(), M3()
        mwman = MiddlewareManager(m1, m2, m3)
        self.assertEqual(mwman.middlewares, (m1, m2, m3))

    def test_enabled_from_settings(self):
        settings = Settings()
        mwman = TestMiddlewareManager.from_settings(settings)
        classes = [x.__class__ for x in mwman.middlewares]
        self.assertEqual(classes, [M1, M3])

class NewName(SomeBaseClass):
    pass


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


class ChunkSize3PickleFifoDiskQueueTest(PickleFifoDiskQueueTest):
    chunksize = 3
