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


class Http11TestCase(HttpTestCase):
    """HTTP 1.1 test case"""
    download_handler_cls = HTTP11DownloadHandler

    def test_download_without_maxsize_limit(self):
        request = Request(self.getURL('file'))
        d = self.download_request(request, Spider('foo'))
        d.addCallback(lambda r: r.body)
        d.addCallback(self.assertEqual, b"0123456789")
        return d

    def test_response_class_choosing_request(self):
        """Tests choosing of correct response type
         in case of Content-Type is empty but body contains text.
        """
        body = b'Some plain text\ndata with tabs\t and null bytes\0'

        def _test_type(response):
            self.assertEqual(type(response), TextResponse)

        request = Request(self.getURL('nocontenttype'), body=body)
        d = self.download_request(request, Spider('foo'))
        d.addCallback(_test_type)
        return d

    @defer.inlineCallbacks
    def test_download_with_maxsize(self):
        request = Request(self.getURL('file'))

        # 10 is minimal size for this request and the limit is only counted on
        # response body. (regardless of headers)
        d = self.download_request(request, Spider('foo', download_maxsize=10))
        d.addCallback(lambda r: r.body)
        d.addCallback(self.assertEqual, b"0123456789")
        yield d

        d = self.download_request(request, Spider('foo', download_maxsize=9))
        yield self.assertFailure(d, defer.CancelledError, error.ConnectionAborted)

    @defer.inlineCallbacks
    def test_download_with_maxsize_very_large_file(self):
        with mock.patch('scrapy.core.downloader.handlers.http11.logger') as logger:
            request = Request(self.getURL('largechunkedfile'))

            def check(logger):
                logger.error.assert_called_once_with(mock.ANY, mock.ANY)

            d = self.download_request(request, Spider('foo', download_maxsize=1500))
            yield self.assertFailure(d, defer.CancelledError, error.ConnectionAborted)

            # As the error message is logged in the dataReceived callback, we
            # have to give a bit of time to the reactor to process the queue
            # after closing the connection.
            d = defer.Deferred()
            d.addCallback(check)
            reactor.callLater(.1, d.callback, logger)
            yield d

    @defer.inlineCallbacks
    def test_download_with_maxsize_per_req(self):
        meta = {'download_maxsize': 2}
        request = Request(self.getURL('file'), meta=meta)
        d = self.download_request(request, Spider('foo'))
        yield self.assertFailure(d, defer.CancelledError, error.ConnectionAborted)

    @defer.inlineCallbacks
    def test_download_with_small_maxsize_per_spider(self):
        request = Request(self.getURL('file'))
        d = self.download_request(request, Spider('foo', download_maxsize=2))
        yield self.assertFailure(d, defer.CancelledError, error.ConnectionAborted)

    def test_download_with_large_maxsize_per_spider(self):
        request = Request(self.getURL('file'))
        d = self.download_request(request, Spider('foo', download_maxsize=100))
        d.addCallback(lambda r: r.body)
        d.addCallback(self.assertEqual, b"0123456789")
        return d

    def test_download_chunked_content(self):
        request = Request(self.getURL('chunked'))
        d = self.download_request(request, Spider('foo'))
        d.addCallback(lambda r: r.body)
        d.addCallback(self.assertEqual, b"chunked content\n")
        return d

    def test_download_broken_content_cause_data_loss(self, url='broken'):
        request = Request(self.getURL(url))
        d = self.download_request(request, Spider('foo'))

        def checkDataLoss(failure):
            if failure.check(ResponseFailed):
                if any(r.check(_DataLoss) for r in failure.value.reasons):
                    return None
            return failure

        d.addCallback(lambda _: self.fail("No DataLoss exception"))
        d.addErrback(checkDataLoss)
        return d

    def test_download_broken_chunked_content_cause_data_loss(self):
        return self.test_download_broken_content_cause_data_loss('broken-chunked')

    def test_download_broken_content_allow_data_loss(self, url='broken'):
        request = Request(self.getURL(url), meta={'download_fail_on_dataloss': False})
        d = self.download_request(request, Spider('foo'))
        d.addCallback(lambda r: r.flags)
        d.addCallback(self.assertEqual, ['dataloss'])
        return d

    def test_download_broken_chunked_content_allow_data_loss(self):
        return self.test_download_broken_content_allow_data_loss('broken-chunked')

    def test_download_broken_content_allow_data_loss_via_setting(self, url='broken'):
        download_handler = self.download_handler_cls(Settings({
            'DOWNLOAD_FAIL_ON_DATALOSS': False,
        }))
        request = Request(self.getURL(url))
        d = download_handler.download_request(request, Spider('foo'))
        d.addCallback(lambda r: r.flags)
        d.addCallback(self.assertEqual, ['dataloss'])
        return d

    def test_download_broken_chunked_content_allow_data_loss_via_setting(self):
        return self.test_download_broken_content_allow_data_loss_via_setting('broken-chunked')


class BinaryIsTextTest(unittest.TestCase):
    def test_binaryistext(self):
        assert binary_is_text(b"hello")

    def test_utf_16_strings_contain_null_bytes(self):
        assert binary_is_text(u"hello".encode('utf-16'))

    def test_one_with_encoding(self):
        assert binary_is_text(b"<div>Price \xa3</div>")

    def test_real_binary_bytes(self):
        assert not binary_is_text(b"\x02\xa3")



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


class ChunkSize2PickleFifoDiskQueueTest(PickleFifoDiskQueueTest):
    chunksize = 2

class BenchCommandTest(CommandTest):

    def test_run(self):
        p = self.proc('bench', '-s', 'LOGSTATS_INTERVAL=0.001',
                '-s', 'CLOSESPIDER_TIMEOUT=0.01')
        log = to_native_str(p.stderr.read())
        self.assertIn('INFO: Crawled', log)
        self.assertNotIn('Unhandled Error', log)

class ImageDownloadCrawlTestCase(FileDownloadCrawlTestCase):
    pipeline_class = 'scrapy.pipelines.images.ImagesPipeline'
    store_setting_key = 'IMAGES_STORE'
    media_key = 'images'
    media_urls_key = 'image_urls'

    # somehow checksums for images are different for Python 3.3
    expected_checksums = None

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


class TestSpider(Spider):
    name = 'test'

    def parse_item(self, response):
        pass

    def handle_error(self, failure):
        pass


class TestOffsiteMiddleware3(TestOffsiteMiddleware2):

    def _get_spider(self):
        return Spider('foo')


class TestRequestMetaPredecence002(MixinNoReferrer, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.NoReferrerWhenDowngradePolicy'}
    req_meta = {'referrer_policy': POLICY_NO_REFERRER}


class DummyDH(object):

    def __init__(self, crawler):
        pass


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

class ProcessorsTest(unittest.TestCase):

    def test_take_first(self):
        proc = TakeFirst()
        self.assertEqual(proc([None, '', 'hello', 'world']), 'hello')
        self.assertEqual(proc([None, '', 0, 'hello', 'world']), 0)

    def test_identity(self):
        proc = Identity()
        self.assertEqual(proc([None, '', 'hello', 'world']),
                         [None, '', 'hello', 'world'])

    def test_join(self):
        proc = Join()
        self.assertRaises(TypeError, proc, [None, '', 'hello', 'world'])
        self.assertEqual(proc(['', 'hello', 'world']), u' hello world')
        self.assertEqual(proc(['hello', 'world']), u'hello world')
        self.assertIsInstance(proc(['hello', 'world']), six.text_type)

    def test_compose(self):
        proc = Compose(lambda v: v[0], str.upper)
        self.assertEqual(proc(['hello', 'world']), 'HELLO')
        proc = Compose(str.upper)
        self.assertEqual(proc(None), None)
        proc = Compose(str.upper, stop_on_none=False)
        self.assertRaises(TypeError, proc, None)

    def test_mapcompose(self):
        filter_world = lambda x: None if x == 'world' else x
        proc = MapCompose(filter_world, six.text_type.upper)
        self.assertEqual(proc([u'hello', u'world', u'this', u'is', u'scrapy']),
                         [u'HELLO', u'THIS', u'IS', u'SCRAPY'])


class TestRequestMetaDefault(MixinDefault, TestRefererMiddleware):
    req_meta = {'referrer_policy': POLICY_SCRAPY_DEFAULT}


class UriResource(resource.Resource):
    """Return the full uri that was requested"""

    def getChild(self, path, request):
        return self

    def render(self, request):
        # Note: this is an ugly hack for CONNECT request timeout test.
        #       Returning some data here fail SSL/TLS handshake
        # ToDo: implement proper HTTPS proxy tests, not faking them.
        if request.method != b'CONNECT':
            return request.uri
        else:
            return b''


class SendCatchLogDeferredTest2(SendCatchLogTest):

    def ok_handler(self, arg, handlers_called):
        handlers_called.add(self.ok_handler)
        assert arg == 'test'
        d = defer.Deferred()
        reactor.callLater(0, d.callback, "OK")
        return d

    def _get_result(self, signal, *a, **kw):
        return send_catch_log_deferred(signal, *a, **kw)

class DecompressionMiddlewareTest(TestCase):

    test_formats = ['tar', 'xml.bz2', 'xml.gz', 'zip']
    uncompressed_body, test_responses = _test_data(test_formats)

    def setUp(self):
        self.mw = DecompressionMiddleware()
        self.spider = Spider('foo')

    def test_known_compression_formats(self):
        for fmt in self.test_formats:
            rsp = self.test_responses[fmt]
            new = self.mw.process_response(None, rsp, self.spider)
            assert isinstance(new, XmlResponse), \
                    'Failed %s, response type %s' % (fmt, type(new).__name__)
            assert_samelines(self, new.body, self.uncompressed_body, fmt)

    def test_plain_response(self):
        rsp = Response(url='http://test.com', body=self.uncompressed_body)
        new = self.mw.process_response(None, rsp, self.spider)
        assert new is rsp
        assert_samelines(self, new.body, rsp.body)

    def test_empty_response(self):
        rsp = Response(url='http://test.com', body=b'')
        new = self.mw.process_response(None, rsp, self.spider)
        assert new is rsp
        assert not rsp.body
        assert not new.body

    def tearDown(self):
        del self.mw


if __name__ == '__main__':
    main()

class M1(object):

    def open_spider(self, spider):
        pass

    def close_spider(self, spider):
        pass

    def process(self, response, request, spider):
        pass

class MiscCommandsTest(CommandTest):

    def test_list(self):
        self.assertEqual(0, self.call('list'))


class BenchCommandTest(CommandTest):

    def test_run(self):
        p = self.proc('bench', '-s', 'LOGSTATS_INTERVAL=0.001',
                '-s', 'CLOSESPIDER_TIMEOUT=0.01')
        log = to_native_str(p.stderr.read())
        self.assertIn('INFO: Crawled', log)
        self.assertNotIn('Unhandled Error', log)

def _mocked_download_func(request, info):
    response = request.meta.get('response')
    return response() if callable(response) else response


class ImagesPipelineTestCase(unittest.TestCase):

    skip = skip

    def setUp(self):
        self.tempdir = mkdtemp()
        self.pipeline = ImagesPipeline(self.tempdir, download_func=_mocked_download_func)

    def tearDown(self):
        rmtree(self.tempdir)

    def test_file_path(self):
        file_path = self.pipeline.file_path
        self.assertEqual(file_path(Request("https://dev.mydeco.com/mydeco.gif")),
                         'full/3fd165099d8e71b8a48b2683946e64dbfad8b52d.jpg')
        self.assertEqual(file_path(Request("http://www.maddiebrown.co.uk///catalogue-items//image_54642_12175_95307.jpg")),
                         'full/0ffcd85d563bca45e2f90becd0ca737bc58a00b2.jpg')
        self.assertEqual(file_path(Request("https://dev.mydeco.com/two/dirs/with%20spaces%2Bsigns.gif")),
                         'full/b250e3a74fff2e4703e310048a5b13eba79379d2.jpg')
        self.assertEqual(file_path(Request("http://www.dfsonline.co.uk/get_prod_image.php?img=status_0907_mdm.jpg")),
                         'full/4507be485f38b0da8a0be9eb2e1dfab8a19223f2.jpg')
        self.assertEqual(file_path(Request("http://www.dorma.co.uk/images/product_details/2532/")),
                         'full/97ee6f8a46cbbb418ea91502fd24176865cf39b2.jpg')
        self.assertEqual(file_path(Request("http://www.dorma.co.uk/images/product_details/2532")),
                         'full/244e0dd7d96a3b7b01f54eded250c9e272577aa1.jpg')
        self.assertEqual(file_path(Request("http://www.dorma.co.uk/images/product_details/2532"),
                                   response=Response("http://www.dorma.co.uk/images/product_details/2532"),
                                   info=object()),
                         'full/244e0dd7d96a3b7b01f54eded250c9e272577aa1.jpg')

    def test_thumbnail_name(self):
        thumb_path = self.pipeline.thumb_path
        name = '50'
        self.assertEqual(thumb_path(Request("file:///tmp/foo.jpg"), name),
                         'thumbs/50/38a86208c36e59d4404db9e37ce04be863ef0335.jpg')
        self.assertEqual(thumb_path(Request("file://foo.png"), name),
                         'thumbs/50/e55b765eba0ec7348e50a1df496040449071b96a.jpg')
        self.assertEqual(thumb_path(Request("file:///tmp/foo"), name),
                         'thumbs/50/0329ad83ebb8e93ea7c7906d46e9ed55f7349a50.jpg')
        self.assertEqual(thumb_path(Request("file:///tmp/some.name/foo"), name),
                         'thumbs/50/850233df65a5b83361798f532f1fc549cd13cbe9.jpg')
        self.assertEqual(thumb_path(Request("file:///tmp/some.name/foo"), name,
                                    response=Response("file:///tmp/some.name/foo"),
                                    info=object()),
                         'thumbs/50/850233df65a5b83361798f532f1fc549cd13cbe9.jpg')

    def test_convert_image(self):
        SIZE = (100, 100)
        # straigh forward case: RGB and JPEG
        COLOUR = (0, 127, 255)
        im = _create_image('JPEG', 'RGB', SIZE, COLOUR)
        converted, _ = self.pipeline.convert_image(im)
        self.assertEqual(converted.mode, 'RGB')
        self.assertEqual(converted.getcolors(), [(10000, COLOUR)])

        # check that thumbnail keep image ratio
        thumbnail, _ = self.pipeline.convert_image(converted, size=(10, 25))
        self.assertEqual(thumbnail.mode, 'RGB')
        self.assertEqual(thumbnail.size, (10, 10))

        # transparency case: RGBA and PNG
        COLOUR = (0, 127, 255, 50)
        im = _create_image('PNG', 'RGBA', SIZE, COLOUR)
        converted, _ = self.pipeline.convert_image(im)
        self.assertEqual(converted.mode, 'RGB')
        self.assertEqual(converted.getcolors(), [(10000, (205, 230, 255))])

        # transparency case with palette: P and PNG
        COLOUR = (0, 127, 255, 50)
        im = _create_image('PNG', 'RGBA', SIZE, COLOUR)
        im = im.convert('P')
        converted, _ = self.pipeline.convert_image(im)
        self.assertEqual(converted.mode, 'RGB')
        self.assertEqual(converted.getcolors(), [(10000, (205, 230, 255))])


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

