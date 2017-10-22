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


class FilesystemStorageTest(DefaultStorageTest):

    storage_class = 'scrapy.extensions.httpcache.FilesystemCacheStorage'

class SitemapSpiderTest(SpiderTest):

    spider_class = SitemapSpider

    BODY = b"SITEMAP"
    f = BytesIO()
    g = gzip.GzipFile(fileobj=f, mode='w+b')
    g.write(BODY)
    g.close()
    GZBODY = f.getvalue()

    def assertSitemapBody(self, response, body):
        spider = self.spider_class("example.com")
        self.assertEqual(spider._get_sitemap_body(response), body)

    def test_get_sitemap_body(self):
        r = XmlResponse(url="http://www.example.com/", body=self.BODY)
        self.assertSitemapBody(r, self.BODY)

        r = HtmlResponse(url="http://www.example.com/", body=self.BODY)
        self.assertSitemapBody(r, None)

        r = Response(url="http://www.example.com/favicon.ico", body=self.BODY)
        self.assertSitemapBody(r, None)

    def test_get_sitemap_body_gzip_headers(self):
        r = Response(url="http://www.example.com/sitemap", body=self.GZBODY,
                     headers={"content-type": "application/gzip"})
        self.assertSitemapBody(r, self.BODY)

    def test_get_sitemap_body_xml_url(self):
        r = TextResponse(url="http://www.example.com/sitemap.xml", body=self.BODY)
        self.assertSitemapBody(r, self.BODY)

    def test_get_sitemap_body_xml_url_compressed(self):
        r = Response(url="http://www.example.com/sitemap.xml.gz", body=self.GZBODY)
        self.assertSitemapBody(r, self.BODY)

        # .xml.gz but body decoded by HttpCompression middleware already
        r = Response(url="http://www.example.com/sitemap.xml.gz", body=self.BODY)
        self.assertSitemapBody(r, self.BODY)

    def test_get_sitemap_urls_from_robotstxt(self):
        robots = b"""# Sitemap files
Sitemap: http://example.com/sitemap.xml
Sitemap: http://example.com/sitemap-product-index.xml
Sitemap: HTTP://example.com/sitemap-uppercase.xml
Sitemap: /sitemap-relative-url.xml
"""

        r = TextResponse(url="http://www.example.com/robots.txt", body=robots)
        spider = self.spider_class("example.com")
        self.assertEqual([req.url for req in spider._parse_sitemap(r)],
                         ['http://example.com/sitemap.xml',
                          'http://example.com/sitemap-product-index.xml',
                          'http://example.com/sitemap-uppercase.xml',
                          'http://www.example.com/sitemap-relative-url.xml'])

    def test_alternate_url_locs(self):
        sitemap = b"""<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:xhtml="http://www.w3.org/1999/xhtml">
        <url>
            <loc>http://www.example.com/english/</loc>
            <xhtml:link rel="alternate" hreflang="de"
                href="http://www.example.com/deutsch/"/>
            <xhtml:link rel="alternate" hreflang="de-ch"
                href="http://www.example.com/schweiz-deutsch/"/>
            <xhtml:link rel="alternate" hreflang="it"
                href="http://www.example.com/italiano/"/>
            <xhtml:link rel="alternate" hreflang="it"/><!-- wrong tag without href -->
        </url>
    </urlset>"""
        r = TextResponse(url="http://www.example.com/sitemap.xml", body=sitemap)
        spider = self.spider_class("example.com")
        self.assertEqual([req.url for req in spider._parse_sitemap(r)],
                         ['http://www.example.com/english/'])

        spider.sitemap_alternate_links = True
        self.assertEqual([req.url for req in spider._parse_sitemap(r)],
                         ['http://www.example.com/english/',
                          'http://www.example.com/deutsch/',
                          'http://www.example.com/schweiz-deutsch/',
                          'http://www.example.com/italiano/'])


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

def _mocked_download_func(request, info):
    response = request.meta.get('response')
    return response() if callable(response) else response


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


class TestOffsiteMiddleware3(TestOffsiteMiddleware2):

    def _get_spider(self):
        return Spider('foo')


class TestItem(Item):
    name = Field()
    age = Field()


class NameItemLoader(ItemLoader):
    default_item_class = TestItem


class ItemTest(unittest.TestCase):

    def assertSortedEqual(self, first, second, msg=None):
        return self.assertEqual(sorted(first), sorted(second), msg)

    def test_simple(self):
        class TestItem(Item):
            name = Field()

        i = TestItem()
        i['name'] = u'name'
        self.assertEqual(i['name'], u'name')

    def test_init(self):
        class TestItem(Item):
            name = Field()

        i = TestItem()
        self.assertRaises(KeyError, i.__getitem__, 'name')

        i2 = TestItem(name=u'john doe')
        self.assertEqual(i2['name'], u'john doe')

        i3 = TestItem({'name': u'john doe'})
        self.assertEqual(i3['name'], u'john doe')

        i4 = TestItem(i3)
        self.assertEqual(i4['name'], u'john doe')

        self.assertRaises(KeyError, TestItem, {'name': u'john doe',
                                               'other': u'foo'})

    def test_invalid_field(self):
        class TestItem(Item):
            pass

        i = TestItem()
        self.assertRaises(KeyError, i.__setitem__, 'field', 'text')
        self.assertRaises(KeyError, i.__getitem__, 'field')

    def test_repr(self):
        class TestItem(Item):
            name = Field()
            number = Field()

        i = TestItem()
        i['name'] = u'John Doe'
        i['number'] = 123
        itemrepr = repr(i)

        if six.PY2:
            self.assertEqual(itemrepr,
                             "{'name': u'John Doe', 'number': 123}")
        else:
            self.assertEqual(itemrepr,
                             "{'name': 'John Doe', 'number': 123}")

        i2 = eval(itemrepr)
        self.assertEqual(i2['name'], 'John Doe')
        self.assertEqual(i2['number'], 123)

    def test_private_attr(self):
        class TestItem(Item):
            name = Field()

        i = TestItem()
        i._private = 'test'
        self.assertEqual(i._private, 'test')

    def test_raise_getattr(self):
        class TestItem(Item):
            name = Field()

        i = TestItem()
        self.assertRaises(AttributeError, getattr, i, 'name')

    def test_raise_setattr(self):
        class TestItem(Item):
            name = Field()

        i = TestItem()
        self.assertRaises(AttributeError, setattr, i, 'name', 'john')

    def test_custom_methods(self):
        class TestItem(Item):
            name = Field()

            def get_name(self):
                return self['name']

            def change_name(self, name):
                self['name'] = name

        i = TestItem()
        self.assertRaises(KeyError, i.get_name)
        i['name'] = u'lala'
        self.assertEqual(i.get_name(), u'lala')
        i.change_name(u'other')
        self.assertEqual(i.get_name(), 'other')

    def test_metaclass(self):
        class TestItem(Item):
            name = Field()
            keys = Field()
            values = Field()

        i = TestItem()
        i['name'] = u'John'
        self.assertEqual(list(i.keys()), ['name'])
        self.assertEqual(list(i.values()), ['John'])

        i['keys'] = u'Keys'
        i['values'] = u'Values'
        self.assertSortedEqual(list(i.keys()), ['keys', 'values', 'name'])
        self.assertSortedEqual(list(i.values()), [u'Keys', u'Values', u'John'])

    def test_metaclass_with_fields_attribute(self):
        class TestItem(Item):
            fields = {'new': Field(default='X')}

        item = TestItem(new=u'New')
        self.assertSortedEqual(list(item.keys()), ['new'])
        self.assertSortedEqual(list(item.values()), [u'New'])

    def test_metaclass_inheritance(self):
        class BaseItem(Item):
            name = Field()
            keys = Field()
            values = Field()

        class TestItem(BaseItem):
            keys = Field()

        i = TestItem()
        i['keys'] = 3
        self.assertEqual(list(i.keys()), ['keys'])
        self.assertEqual(list(i.values()), [3])

    def test_metaclass_multiple_inheritance_simple(self):
        class A(Item):
            fields = {'load': Field(default='A')}
            save = Field(default='A')

        class B(A): pass

        class C(Item):
            fields = {'load': Field(default='C')}
            save = Field(default='C')

        class D(B, C): pass

        item = D(save='X', load='Y')
        self.assertEqual(item['save'], 'X')
        self.assertEqual(item['load'], 'Y')
        self.assertEqual(D.fields, {'load': {'default': 'A'},
            'save': {'default': 'A'}})

        # D class inverted
        class E(C, B): pass

        self.assertEqual(E(save='X')['save'], 'X')
        self.assertEqual(E(load='X')['load'], 'X')
        self.assertEqual(E.fields, {'load': {'default': 'C'},
            'save': {'default': 'C'}})

    def test_metaclass_multiple_inheritance_diamond(self):
        class A(Item):
            fields = {'update': Field(default='A')}
            save = Field(default='A')
            load = Field(default='A')

        class B(A): pass

        class C(A):
            fields = {'update': Field(default='C')}
            save = Field(default='C')

        class D(B, C):
            fields = {'update': Field(default='D')}
            load = Field(default='D')

        self.assertEqual(D(save='X')['save'], 'X')
        self.assertEqual(D(load='X')['load'], 'X')
        self.assertEqual(D.fields, {'save': {'default': 'C'},
            'load': {'default': 'D'}, 'update': {'default': 'D'}})

        # D class inverted
        class E(C, B):
            load = Field(default='E')

        self.assertEqual(E(save='X')['save'], 'X')
        self.assertEqual(E(load='X')['load'], 'X')
        self.assertEqual(E.fields, {'save': {'default': 'C'},
            'load': {'default': 'E'}, 'update': {'default': 'C'}})

    def test_metaclass_multiple_inheritance_without_metaclass(self):
        class A(Item):
            fields = {'load': Field(default='A')}
            save = Field(default='A')

        class B(A): pass

        class C(object):
            fields = {'load': Field(default='C')}
            not_allowed = Field(default='not_allowed')
            save = Field(default='C')

        class D(B, C): pass

        self.assertRaises(KeyError, D, not_allowed='value')
        self.assertEqual(D(save='X')['save'], 'X')
        self.assertEqual(D.fields, {'save': {'default': 'A'},
            'load': {'default': 'A'}})

        # D class inverted
        class E(C, B): pass

        self.assertRaises(KeyError, E, not_allowed='value')
        self.assertEqual(E(save='X')['save'], 'X')
        self.assertEqual(E.fields, {'save': {'default': 'A'},
            'load': {'default': 'A'}})

    def test_to_dict(self):
        class TestItem(Item):
            name = Field()

        i = TestItem()
        i['name'] = u'John'
        self.assertEqual(dict(i), {'name': u'John'})

    def test_copy(self):
        class TestItem(Item):
            name = Field()
        item = TestItem({'name':'lower'})
        copied_item = item.copy()
        self.assertNotEqual(id(item), id(copied_item))
        copied_item['name'] = copied_item['name'].upper()
        self.assertNotEqual(item['name'], copied_item['name'])


class RFPDupeFilterTest(unittest.TestCase):

    def test_filter(self):
        dupefilter = RFPDupeFilter()
        dupefilter.open()

        r1 = Request('http://scrapytest.org/1')
        r2 = Request('http://scrapytest.org/2')
        r3 = Request('http://scrapytest.org/2')

        assert not dupefilter.request_seen(r1)
        assert dupefilter.request_seen(r1)

        assert not dupefilter.request_seen(r2)
        assert dupefilter.request_seen(r3)

        dupefilter.close('finished')

    def test_dupefilter_path(self):
        r1 = Request('http://scrapytest.org/1')
        r2 = Request('http://scrapytest.org/2')

        path = tempfile.mkdtemp()
        try:
            df = RFPDupeFilter(path)
            df.open()
            assert not df.request_seen(r1)
            assert df.request_seen(r1)
            df.close('finished')

            df2 = RFPDupeFilter(path)
            df2.open()
            assert df2.request_seen(r1)
            assert not df2.request_seen(r2)
            assert df2.request_seen(r2)
            df2.close('finished')
        finally:
            shutil.rmtree(path)

    def test_request_fingerprint(self):
        """Test if customization of request_fingerprint method will change
        output of request_seen.

        """
        r1 = Request('http://scrapytest.org/index.html')
        r2 = Request('http://scrapytest.org/INDEX.html')

        dupefilter = RFPDupeFilter()
        dupefilter.open()

        assert not dupefilter.request_seen(r1)
        assert not dupefilter.request_seen(r2)

        dupefilter.close('finished')

        class CaseInsensitiveRFPDupeFilter(RFPDupeFilter):

            def request_fingerprint(self, request):
                fp = hashlib.sha1()
                fp.update(to_bytes(request.url.lower()))
                return fp.hexdigest()

        case_insensitive_dupefilter = CaseInsensitiveRFPDupeFilter()
        case_insensitive_dupefilter.open()

        assert not case_insensitive_dupefilter.request_seen(r1)
        assert case_insensitive_dupefilter.request_seen(r2)

        case_insensitive_dupefilter.close('finished')

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


class TestReferrerOnRedirectStrictOrigin(TestReferrerOnRedirect):
    """
    Strict Origin policy will always send the "origin" as referrer
    (think of it as the parent URL without the path part),
    unless the security level is lower and no "Referer" is sent.

    Redirections from secure to non-secure URLs should have the
    "Referrer" header removed if necessary.
    """
    settings = {'REFERRER_POLICY': POLICY_STRICT_ORIGIN}
    scenarii = [
        (   'http://scrapytest.org/101',
            'http://scrapytest.org/102',
            (
                (301, 'http://scrapytest.org/103'),
                (301, 'http://scrapytest.org/104'),
            ),
            b'http://scrapytest.org/',  # send origin
            b'http://scrapytest.org/',  # redirects to same origin: send origin
        ),
        (   'https://scrapytest.org/201',
            'https://scrapytest.org/202',
            (
                # redirecting to non-secure URL: no referrer
                (301, 'http://scrapytest.org/203'),
            ),
            b'https://scrapytest.org/',
            None,
        ),
        (   'https://scrapytest.org/301',
            'https://scrapytest.org/302',
            (
                # redirecting to non-secure URL (different domain): no referrer
                (301, 'http://example.com/303'),
            ),
            b'https://scrapytest.org/',
            None,
        ),
        (   'http://scrapy.org/401',
            'http://example.com/402',
            (
                (301, 'http://scrapytest.org/403'),
            ),
            b'http://scrapy.org/',
            b'http://scrapy.org/',
        ),
        (   'https://scrapy.org/501',
            'https://example.com/502',
            (
                # HTTPS all along, so origin referrer is kept as-is
                (301, 'https://google.com/503'),
                (301, 'https://facebook.com/504'),
            ),
            b'https://scrapy.org/',
            b'https://scrapy.org/',
        ),
        (   'https://scrapytest.org/601',
            'http://scrapytest.org/602',                # TLS to non-TLS: no referrer
            (
                (301, 'https://scrapytest.org/603'),    # TLS URL again: (still) no referrer
            ),
            None,
            None,
        ),
    ]


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


class TestItem(NameItem):
    url = Field()
    summary = Field()


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

class MySpider(scrapy.Spider):
    name = '{0}'

    def parse(self, response):
        if getattr(self, 'test_arg', None):
            self.logger.debug('It Works!')
        return [scrapy.Item(), dict(foo='bar')]


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

class RunSpiderCommandTest(CommandTest):

    debug_log_spider = """
import scrapy

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


class ToUnicodeTest(unittest.TestCase):
    def test_converting_an_utf8_encoded_string_to_unicode(self):
        self.assertEqual(to_unicode(b'lel\xc3\xb1e'), u'lel\xf1e')

    def test_converting_a_latin_1_encoded_string_to_unicode(self):
        self.assertEqual(to_unicode(b'lel\xf1e', 'latin-1'), u'lel\xf1e')

    def test_converting_a_unicode_to_unicode_should_return_the_same_object(self):
        self.assertEqual(to_unicode(u'\xf1e\xf1e\xf1e'), u'\xf1e\xf1e\xf1e')

    def test_converting_a_strange_object_should_raise_TypeError(self):
        self.assertRaises(TypeError, to_unicode, 423)

    def test_errors_argument(self):
        self.assertEqual(
            to_unicode(b'a\xedb', 'utf-8', errors='replace'),
            u'a\ufffdb'
        )


class EncodingResource(resource.Resource):
    out_encoding = 'cp1251'

    def render(self, request):
        body = to_unicode(request.content.read())
        request.setHeader(b'content-encoding', self.out_encoding)
        return body.encode(self.out_encoding)


class SendCatchLogDeferredTest2(SendCatchLogTest):

    def ok_handler(self, arg, handlers_called):
        handlers_called.add(self.ok_handler)
        assert arg == 'test'
        d = defer.Deferred()
        reactor.callLater(0, d.callback, "OK")
        return d

    def _get_result(self, signal, *a, **kw):
        return send_catch_log_deferred(signal, *a, **kw)

class TestRequestMetaPredecence003(MixinUnsafeUrl, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.OriginWhenCrossOriginPolicy'}
    req_meta = {'referrer_policy': POLICY_UNSAFE_URL}


class TestReferrerOnRedirectStrictOrigin(TestReferrerOnRedirect):
    """
    Strict Origin policy will always send the "origin" as referrer
    (think of it as the parent URL without the path part),
    unless the security level is lower and no "Referer" is sent.

    Redirections from secure to non-secure URLs should have the
    "Referrer" header removed if necessary.
    """
    settings = {'REFERRER_POLICY': POLICY_STRICT_ORIGIN}
    scenarii = [
        (   'http://scrapytest.org/101',
            'http://scrapytest.org/102',
            (
                (301, 'http://scrapytest.org/103'),
                (301, 'http://scrapytest.org/104'),
            ),
            b'http://scrapytest.org/',  # send origin
            b'http://scrapytest.org/',  # redirects to same origin: send origin
        ),
        (   'https://scrapytest.org/201',
            'https://scrapytest.org/202',
            (
                # redirecting to non-secure URL: no referrer
                (301, 'http://scrapytest.org/203'),
            ),
            b'https://scrapytest.org/',
            None,
        ),
        (   'https://scrapytest.org/301',
            'https://scrapytest.org/302',
            (
                # redirecting to non-secure URL (different domain): no referrer
                (301, 'http://example.com/303'),
            ),
            b'https://scrapytest.org/',
            None,
        ),
        (   'http://scrapy.org/401',
            'http://example.com/402',
            (
                (301, 'http://scrapytest.org/403'),
            ),
            b'http://scrapy.org/',
            b'http://scrapy.org/',
        ),
        (   'https://scrapy.org/501',
            'https://example.com/502',
            (
                # HTTPS all along, so origin referrer is kept as-is
                (301, 'https://google.com/503'),
                (301, 'https://facebook.com/504'),
            ),
            b'https://scrapy.org/',
            b'https://scrapy.org/',
        ),
        (   'https://scrapytest.org/601',
            'http://scrapytest.org/602',                # TLS to non-TLS: no referrer
            (
                (301, 'https://scrapytest.org/603'),    # TLS URL again: (still) no referrer
            ),
            None,
            None,
        ),
    ]


def _mocked_download_func(request, info):
    response = request.meta.get('response')
    return response() if callable(response) else response


class SpiderSettingsTestCase(unittest.TestCase):
    def test_spider_custom_settings(self):
        class MySpider(scrapy.Spider):
            name = 'spider'
            custom_settings = {
                'AUTOTHROTTLE_ENABLED': True
            }

        crawler = Crawler(MySpider, {})
        enabled_exts = [e.__class__ for e in crawler.extensions.middlewares]
        self.assertIn(AutoThrottle, enabled_exts)


class UserAgentMiddlewareTest(TestCase):

    def get_spider_and_mw(self, default_useragent):
        crawler = get_crawler(Spider, {'USER_AGENT': default_useragent})
        spider = crawler._create_spider('foo')
        return spider, UserAgentMiddleware.from_crawler(crawler)

    def test_default_agent(self):
        spider, mw = self.get_spider_and_mw('default_useragent')
        req = Request('http://scrapytest.org/')
        assert mw.process_request(req, spider) is None
        self.assertEqual(req.headers['User-Agent'], b'default_useragent')

    def test_remove_agent(self):
        # settings UESR_AGENT to None should remove the user agent
        spider, mw = self.get_spider_and_mw('default_useragent')
        spider.user_agent = None
        mw.spider_opened(spider)
        req = Request('http://scrapytest.org/')
        assert mw.process_request(req, spider) is None
        assert req.headers.get('User-Agent') is None

    def test_spider_agent(self):
        spider, mw = self.get_spider_and_mw('default_useragent')
        spider.user_agent = 'spider_useragent'
        mw.spider_opened(spider)
        req = Request('http://scrapytest.org/')
        assert mw.process_request(req, spider) is None
        self.assertEqual(req.headers['User-Agent'], b'spider_useragent')

    def test_header_agent(self):
        spider, mw = self.get_spider_and_mw('default_useragent')
        spider.user_agent = 'spider_useragent'
        mw.spider_opened(spider)
        req = Request('http://scrapytest.org/',
                      headers={'User-Agent': 'header_useragent'})
        assert mw.process_request(req, spider) is None
        self.assertEqual(req.headers['User-Agent'], b'header_useragent')

    def test_no_agent(self):
        spider, mw = self.get_spider_and_mw(None)
        spider.user_agent = None
        mw.spider_opened(spider)
        req = Request('http://scrapytest.org/')
        assert mw.process_request(req, spider) is None
        assert 'User-Agent' not in req.headers

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

