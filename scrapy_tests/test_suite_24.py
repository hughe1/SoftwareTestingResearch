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


class TestMiddlewareManager(MiddlewareManager):

    @classmethod
    def _get_mwlist_from_settings(cls, settings):
        return ['tests.test_middleware.%s' % x for x in ['M1', 'MOff', 'M3']]

    def _add_middleware(self, mw):
        super(TestMiddlewareManager, self)._add_middleware(mw)
        if hasattr(mw, 'process'):
            self.methods['process'].append(mw.process)

class CustomSpiderLoader(SpiderLoader):
    pass


class TestSettingsOrigin(MixinOrigin, TestRefererMiddleware):
    settings = {'REFERRER_POLICY': 'scrapy.spidermiddlewares.referer.OriginPolicy'}


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

class StripUrl(unittest.TestCase):

    def test_noop(self):
        self.assertEqual(strip_url(
            'http://www.example.com/index.html'),
            'http://www.example.com/index.html')

    def test_noop_query_string(self):
        self.assertEqual(strip_url(
            'http://www.example.com/index.html?somekey=somevalue'),
            'http://www.example.com/index.html?somekey=somevalue')

    def test_fragments(self):
        self.assertEqual(strip_url(
            'http://www.example.com/index.html?somekey=somevalue#section', strip_fragment=False),
            'http://www.example.com/index.html?somekey=somevalue#section')

    def test_path(self):
        for input_url, origin, output_url in [
            ('http://www.example.com/',
             False,
             'http://www.example.com/'),

            ('http://www.example.com',
             False,
             'http://www.example.com'),

            ('http://www.example.com',
             True,
             'http://www.example.com/'),
            ]:
            self.assertEqual(strip_url(input_url, origin_only=origin), output_url)

    def test_credentials(self):
        for i, o in [
            ('http://username@www.example.com/index.html?somekey=somevalue#section',
             'http://www.example.com/index.html?somekey=somevalue'),

            ('https://username:@www.example.com/index.html?somekey=somevalue#section',
             'https://www.example.com/index.html?somekey=somevalue'),

            ('ftp://username:password@www.example.com/index.html?somekey=somevalue#section',
             'ftp://www.example.com/index.html?somekey=somevalue'),
            ]:
            self.assertEqual(strip_url(i, strip_credentials=True), o)

    def test_credentials_encoded_delims(self):
        for i, o in [
            # user: "username@"
            # password: none
            ('http://username%40@www.example.com/index.html?somekey=somevalue#section',
             'http://www.example.com/index.html?somekey=somevalue'),

            # user: "username:pass"
            # password: ""
            ('https://username%3Apass:@www.example.com/index.html?somekey=somevalue#section',
             'https://www.example.com/index.html?somekey=somevalue'),

            # user: "me"
            # password: "user@domain.com"
            ('ftp://me:user%40domain.com@www.example.com/index.html?somekey=somevalue#section',
             'ftp://www.example.com/index.html?somekey=somevalue'),
            ]:
            self.assertEqual(strip_url(i, strip_credentials=True), o)

    def test_default_ports_creds_off(self):
        for i, o in [
            ('http://username:password@www.example.com:80/index.html?somekey=somevalue#section',
             'http://www.example.com/index.html?somekey=somevalue'),

            ('http://username:password@www.example.com:8080/index.html#section',
             'http://www.example.com:8080/index.html'),

            ('http://username:password@www.example.com:443/index.html?somekey=somevalue&someotherkey=sov#section',
             'http://www.example.com:443/index.html?somekey=somevalue&someotherkey=sov'),

            ('https://username:password@www.example.com:443/index.html',
             'https://www.example.com/index.html'),

            ('https://username:password@www.example.com:442/index.html',
             'https://www.example.com:442/index.html'),

            ('https://username:password@www.example.com:80/index.html',
             'https://www.example.com:80/index.html'),

            ('ftp://username:password@www.example.com:21/file.txt',
             'ftp://www.example.com/file.txt'),

            ('ftp://username:password@www.example.com:221/file.txt',
             'ftp://www.example.com:221/file.txt'),
            ]:
            self.assertEqual(strip_url(i), o)

    def test_default_ports(self):
        for i, o in [
            ('http://username:password@www.example.com:80/index.html',
             'http://username:password@www.example.com/index.html'),

            ('http://username:password@www.example.com:8080/index.html',
             'http://username:password@www.example.com:8080/index.html'),

            ('http://username:password@www.example.com:443/index.html',
             'http://username:password@www.example.com:443/index.html'),

            ('https://username:password@www.example.com:443/index.html',
             'https://username:password@www.example.com/index.html'),

            ('https://username:password@www.example.com:442/index.html',
             'https://username:password@www.example.com:442/index.html'),

            ('https://username:password@www.example.com:80/index.html',
             'https://username:password@www.example.com:80/index.html'),

            ('ftp://username:password@www.example.com:21/file.txt',
             'ftp://username:password@www.example.com/file.txt'),

            ('ftp://username:password@www.example.com:221/file.txt',
             'ftp://username:password@www.example.com:221/file.txt'),
            ]:
            self.assertEqual(strip_url(i, strip_default_port=True, strip_credentials=False), o)

    def test_default_ports_keep(self):
        for i, o in [
            ('http://username:password@www.example.com:80/index.html?somekey=somevalue&someotherkey=sov#section',
             'http://username:password@www.example.com:80/index.html?somekey=somevalue&someotherkey=sov'),

            ('http://username:password@www.example.com:8080/index.html?somekey=somevalue&someotherkey=sov#section',
             'http://username:password@www.example.com:8080/index.html?somekey=somevalue&someotherkey=sov'),

            ('http://username:password@www.example.com:443/index.html',
             'http://username:password@www.example.com:443/index.html'),

            ('https://username:password@www.example.com:443/index.html',
             'https://username:password@www.example.com:443/index.html'),

            ('https://username:password@www.example.com:442/index.html',
             'https://username:password@www.example.com:442/index.html'),

            ('https://username:password@www.example.com:80/index.html',
             'https://username:password@www.example.com:80/index.html'),

            ('ftp://username:password@www.example.com:21/file.txt',
             'ftp://username:password@www.example.com:21/file.txt'),

            ('ftp://username:password@www.example.com:221/file.txt',
             'ftp://username:password@www.example.com:221/file.txt'),
            ]:
            self.assertEqual(strip_url(i, strip_default_port=False, strip_credentials=False), o)

    def test_origin_only(self):
        for i, o in [
            ('http://username:password@www.example.com/index.html',
             'http://www.example.com/'),

            ('http://username:password@www.example.com:80/foo/bar?query=value#somefrag',
             'http://www.example.com/'),

            ('http://username:password@www.example.com:8008/foo/bar?query=value#somefrag',
             'http://www.example.com:8008/'),

            ('https://username:password@www.example.com:443/index.html',
             'https://www.example.com/'),
            ]:
            self.assertEqual(strip_url(i, origin_only=True), o)


if __name__ == "__main__":
    unittest.main()

class ResponseMock(object):
    url = 'http://scrapy.org'


class SelectorTestCase(unittest.TestCase):

    def test_simple_selection(self):
        """Simple selector tests"""
        body = b"<p><input name='a'value='1'/><input name='b'value='2'/></p>"
        response = TextResponse(url="http://example.com", body=body, encoding='utf-8')
        sel = Selector(response)

        xl = sel.xpath('//input')
        self.assertEqual(2, len(xl))
        for x in xl:
            assert isinstance(x, Selector)

        self.assertEqual(sel.xpath('//input').extract(),
                         [x.extract() for x in sel.xpath('//input')])

        self.assertEqual([x.extract() for x in sel.xpath("//input[@name='a']/@name")],
                         [u'a'])
        self.assertEqual([x.extract() for x in sel.xpath("number(concat(//input[@name='a']/@value, //input[@name='b']/@value))")],
                         [u'12.0'])

        self.assertEqual(sel.xpath("concat('xpath', 'rules')").extract(),
                         [u'xpathrules'])
        self.assertEqual([x.extract() for x in sel.xpath("concat(//input[@name='a']/@value, //input[@name='b']/@value)")],
                         [u'12'])

    def test_root_base_url(self):
        body = b'<html><form action="/path"><input name="a" /></form></html>'
        url = "http://example.com"
        response = TextResponse(url=url, body=body, encoding='utf-8')
        sel = Selector(response)
        self.assertEqual(url, sel.root.base)

    def test_deprecated_root_argument(self):
        with warnings.catch_warnings(record=True) as w:
            root = etree.fromstring(u'<html/>')
            sel = Selector(_root=root)
            self.assertIs(root, sel.root)
            self.assertEqual(str(w[-1].message),
                             'Argument `_root` is deprecated, use `root` instead')

    def test_deprecated_root_argument_ambiguous(self):
        with warnings.catch_warnings(record=True) as w:
            _root = etree.fromstring(u'<xml/>')
            root = etree.fromstring(u'<html/>')
            sel = Selector(_root=_root, root=root)
            self.assertIs(root, sel.root)
            self.assertIn('Ignoring deprecated `_root` argument', str(w[-1].message))

    def test_flavor_detection(self):
        text = b'<div><img src="a.jpg"><p>Hello</div>'
        sel = Selector(XmlResponse('http://example.com', body=text, encoding='utf-8'))
        self.assertEqual(sel.type, 'xml')
        self.assertEqual(sel.xpath("//div").extract(),
                         [u'<div><img src="a.jpg"><p>Hello</p></img></div>'])

        sel = Selector(HtmlResponse('http://example.com', body=text, encoding='utf-8'))
        self.assertEqual(sel.type, 'html')
        self.assertEqual(sel.xpath("//div").extract(),
                         [u'<div><img src="a.jpg"><p>Hello</p></div>'])

    def test_http_header_encoding_precedence(self):
        # u'\xa3'     = pound symbol in unicode
        # u'\xc2\xa3' = pound symbol in utf-8
        # u'\xa3'     = pound symbol in latin-1 (iso-8859-1)

        meta = u'<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">'
        head = u'<head>' + meta + u'</head>'
        body_content = u'<span id="blank">\xa3</span>'
        body = u'<body>' + body_content + u'</body>'
        html = u'<html>' + head + body + u'</html>'
        encoding = 'utf-8'
        html_utf8 = html.encode(encoding)

        headers = {'Content-Type': ['text/html; charset=utf-8']}
        response = HtmlResponse(url="http://example.com", headers=headers, body=html_utf8)
        x = Selector(response)
        self.assertEqual(x.xpath("//span[@id='blank']/text()").extract(),
                          [u'\xa3'])

    def test_badly_encoded_body(self):
        # \xe9 alone isn't valid utf8 sequence
        r1 = TextResponse('http://www.example.com', \
                          body=b'<html><p>an Jos\xe9 de</p><html>', \
                          encoding='utf-8')
        Selector(r1).xpath('//text()').extract()

    def test_weakref_slots(self):
        """Check that classes are using slots and are weak-referenceable"""
        x = Selector(text='')
        weakref.ref(x)
        assert not hasattr(x, '__dict__'), "%s does not use __slots__" % \
            x.__class__.__name__

    def test_deprecated_selector_methods(self):
        sel = Selector(TextResponse(url="http://example.com", body=b'<p>some text</p>'))

        with warnings.catch_warnings(record=True) as w:
            sel.select('//p')
            self.assertSubstring('Use .xpath() instead', str(w[-1].message))

        with warnings.catch_warnings(record=True) as w:
            sel.extract_unquoted()
            self.assertSubstring('Use .extract() instead', str(w[-1].message))

    def test_deprecated_selectorlist_methods(self):
        sel = Selector(TextResponse(url="http://example.com", body=b'<p>some text</p>'))

        with warnings.catch_warnings(record=True) as w:
            sel.xpath('//p').select('.')
            self.assertSubstring('Use .xpath() instead', str(w[-1].message))

        with warnings.catch_warnings(record=True) as w:
            sel.xpath('//p').extract_unquoted()
            self.assertSubstring('Use .extract() instead', str(w[-1].message))

    def test_selector_bad_args(self):
        with self.assertRaisesRegexp(ValueError, 'received both response and text'):
            Selector(TextResponse(url='http://example.com', body=b''), text=u'')


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

class StdoutFeedStorageTest(unittest.TestCase):

    @defer.inlineCallbacks
    def test_store(self):
        out = BytesIO()
        storage = StdoutFeedStorage('stdout:', _stdout=out)
        file = storage.open(scrapy.Spider("default"))
        file.write(b"content")
        yield storage.store(file)
        self.assertEqual(out.getvalue(), b"content")


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


class LeveldbStorageTest(DefaultStorageTest):

    pytest.importorskip('leveldb')
    storage_class = 'scrapy.extensions.httpcache.LeveldbCacheStorage'


class TestDownloaderStats(TestCase):

    def setUp(self):
        self.crawler = get_crawler(Spider)
        self.spider = self.crawler._create_spider('scrapytest.org')
        self.mw = DownloaderStats(self.crawler.stats)

        self.crawler.stats.open_spider(self.spider)

        self.req = Request('http://scrapytest.org')
        self.res = Response('scrapytest.org', status=400)

    def assertStatsEqual(self, key, value):
        self.assertEqual(
            self.crawler.stats.get_value(key, spider=self.spider),
            value,
            str(self.crawler.stats.get_stats(self.spider))
        )

    def test_process_request(self):
        self.mw.process_request(self.req, self.spider)
        self.assertStatsEqual('downloader/request_count', 1)

    def test_process_response(self):
        self.mw.process_response(self.req, self.res, self.spider)
        self.assertStatsEqual('downloader/response_count', 1)

    def test_process_exception(self):
        self.mw.process_exception(self.req, MyException(), self.spider)
        self.assertStatsEqual('downloader/exception_count', 1)
        self.assertStatsEqual(
            'downloader/exception_type_count/tests.test_downloadermiddleware_stats.MyException',
            1
        )

    def tearDown(self):
        self.crawler.stats.close_spider(self.spider, '')

class CrawlerRun(object):
    """A class to run the crawler and keep track of events occurred"""

    def __init__(self, spider_class):
        self.spider = None
        self.respplug = []
        self.reqplug = []
        self.reqdropped = []
        self.itemresp = []
        self.signals_catched = {}
        self.spider_class = spider_class

    def run(self):
        self.port = start_test_site()
        self.portno = self.port.getHost().port

        start_urls = [self.geturl("/"), self.geturl("/redirect"),
                      self.geturl("/redirect")]  # a duplicate

        for name, signal in vars(signals).items():
            if not name.startswith('_'):
                dispatcher.connect(self.record_signal, signal)

        self.crawler = get_crawler(self.spider_class)
        self.crawler.signals.connect(self.item_scraped, signals.item_scraped)
        self.crawler.signals.connect(self.request_scheduled, signals.request_scheduled)
        self.crawler.signals.connect(self.request_dropped, signals.request_dropped)
        self.crawler.signals.connect(self.response_downloaded, signals.response_downloaded)
        self.crawler.crawl(start_urls=start_urls)
        self.spider = self.crawler.spider

        self.deferred = defer.Deferred()
        dispatcher.connect(self.stop, signals.engine_stopped)
        return self.deferred

    def stop(self):
        self.port.stopListening()
        for name, signal in vars(signals).items():
            if not name.startswith('_'):
                disconnect_all(signal)
        self.deferred.callback(None)

    def geturl(self, path):
        return "http://localhost:%s%s" % (self.portno, path)

    def getpath(self, url):
        u = urlparse(url)
        return u.path

    def item_scraped(self, item, spider, response):
        self.itemresp.append((item, response))

    def request_scheduled(self, request, spider):
        self.reqplug.append((request, spider))

    def request_dropped(self, request, spider):
        self.reqdropped.append((request, spider))

    def response_downloaded(self, response, spider):
        self.respplug.append((response, spider))

    def record_signal(self, *args, **kwargs):
        """Record a signal and its parameters"""
        signalargs = kwargs.copy()
        sig = signalargs.pop('signal')
        signalargs.pop('sender', None)
        self.signals_catched[sig] = signalargs


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


class FilesystemStorageGzipTest(FilesystemStorageTest):

    def _get_settings(self, **new_settings):
        new_settings.setdefault('HTTPCACHE_GZIP', True)
        return super(FilesystemStorageTest, self)._get_settings(**new_settings)

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


class EmptyContentTypeHeaderResource(resource.Resource):
    """
    A testing resource which renders itself as the value of request body
    without content-type header in response.
    """
    def render(self, request):
        request.setHeader("content-type", "")
        return request.content.read()


class ChunkSize2MarshalFifoDiskQueueTest(MarshalFifoDiskQueueTest):
    chunksize = 2

def _mocked_download_func(request, info):
    response = request.meta.get('response')
    return response() if callable(response) else response


class AnonymousFTPTestCase(BaseFTPTestCase):

    username = "anonymous"
    req_meta = {}

    def setUp(self):
        from twisted.protocols.ftp import FTPRealm, FTPFactory
        from scrapy.core.downloader.handlers.ftp import FTPDownloadHandler

        # setup dir and test file
        self.directory = self.mktemp()
        os.mkdir(self.directory)

        fp = FilePath(self.directory)
        fp.child('file.txt').setContent(b"I have the power!")
        fp.child('file with spaces.txt').setContent(b"Moooooooooo power!")

        # setup server for anonymous access
        realm = FTPRealm(anonymousRoot=self.directory)
        p = portal.Portal(realm)
        p.registerChecker(checkers.AllowAnonymousAccess(),
                          credentials.IAnonymous)

        self.factory = FTPFactory(portal=p,
                                  userAnonymous=self.username)
        self.port = reactor.listenTCP(0, self.factory, interface="127.0.0.1")
        self.portNum = self.port.getHost().port
        self.download_handler = FTPDownloadHandler(Settings())
        self.addCleanup(self.port.stopListening)

    def tearDown(self):
        shutil.rmtree(self.directory)


class DeprecatedClassesTest(unittest.TestCase):

    def test_deprecated_warnings(self):
        for cls in [ScrapyHTMLTranslator, ScrapyGenericTranslator, ScrapyXPathExpr]:
            with warnings.catch_warnings(record=True) as w:
                obj = cls()
                self.assertIn('%s is deprecated' % cls.__name__, str(w[-1].message),
                              'Missing deprecate warning for %s' % cls.__name__)



def _test_data(formats):
    uncompressed_body = get_testdata('compressed', 'feed-sample1.xml')
    test_responses = {}
    for format in formats:
        body = get_testdata('compressed', 'feed-sample1.' + format)
        test_responses[format] = Response('http://foo.com/bar', body=body)
    return uncompressed_body, test_responses


class CustomItem(Item):

    name = Field()

    def __str__(self):
        return "name: %s" % self['name']


class CookiesMiddlewareTest(TestCase):

    def assertCookieValEqual(self, first, second, msg=None):
        cookievaleq = lambda cv: re.split(';\s*', cv.decode('latin1'))
        return self.assertEqual(
            sorted(cookievaleq(first)),
            sorted(cookievaleq(second)), msg)

    def setUp(self):
        self.spider = Spider('foo')
        self.mw = CookiesMiddleware()

    def tearDown(self):
        del self.mw

    def test_basic(self):
        req = Request('http://scrapytest.org/')
        assert self.mw.process_request(req, self.spider) is None
        assert 'Cookie' not in req.headers

        headers = {'Set-Cookie': 'C1=value1; path=/'}
        res = Response('http://scrapytest.org/', headers=headers)
        assert self.mw.process_response(req, res, self.spider) is res

        req2 = Request('http://scrapytest.org/sub1/')
        assert self.mw.process_request(req2, self.spider) is None
        self.assertEqual(req2.headers.get('Cookie'), b"C1=value1")

    def test_setting_false_cookies_enabled(self):
        self.assertRaises(
            NotConfigured,
            CookiesMiddleware.from_crawler,
            get_crawler(settings_dict={'COOKIES_ENABLED': False})
        )

    def test_setting_default_cookies_enabled(self):
        self.assertIsInstance(
            CookiesMiddleware.from_crawler(get_crawler()),
            CookiesMiddleware
        )

    def test_setting_true_cookies_enabled(self):
        self.assertIsInstance(
            CookiesMiddleware.from_crawler(
                get_crawler(settings_dict={'COOKIES_ENABLED': True})
            ),
            CookiesMiddleware
        )

    def test_setting_enabled_cookies_debug(self):
        crawler = get_crawler(settings_dict={'COOKIES_DEBUG': True})
        mw = CookiesMiddleware.from_crawler(crawler)
        with LogCapture('scrapy.downloadermiddlewares.cookies',
                        propagate=False,
                        level=logging.DEBUG) as l:
            req = Request('http://scrapytest.org/')
            res = Response('http://scrapytest.org/',
                           headers={'Set-Cookie': 'C1=value1; path=/'})
            mw.process_response(req, res, crawler.spider)
            req2 = Request('http://scrapytest.org/sub1/')
            mw.process_request(req2, crawler.spider)

            l.check(
                ('scrapy.downloadermiddlewares.cookies',
                 'DEBUG',
                 'Received cookies from: <200 http://scrapytest.org/>\n'
                 'Set-Cookie: C1=value1; path=/\n'),
                ('scrapy.downloadermiddlewares.cookies',
                 'DEBUG',
                 'Sending cookies to: <GET http://scrapytest.org/sub1/>\n'
                 'Cookie: C1=value1\n'),
            )

    def test_setting_disabled_cookies_debug(self):
        crawler = get_crawler(settings_dict={'COOKIES_DEBUG': False})
        mw = CookiesMiddleware.from_crawler(crawler)
        with LogCapture('scrapy.downloadermiddlewares.cookies',
                        propagate=False,
                        level=logging.DEBUG) as l:
            req = Request('http://scrapytest.org/')
            res = Response('http://scrapytest.org/',
                           headers={'Set-Cookie': 'C1=value1; path=/'})
            mw.process_response(req, res, crawler.spider)
            req2 = Request('http://scrapytest.org/sub1/')
            mw.process_request(req2, crawler.spider)

            l.check()

    def test_do_not_break_on_non_utf8_header(self):
        req = Request('http://scrapytest.org/')
        assert self.mw.process_request(req, self.spider) is None
        assert 'Cookie' not in req.headers

        headers = {'Set-Cookie': b'C1=in\xa3valid; path=/',
                   'Other': b'ignore\xa3me'}
        res = Response('http://scrapytest.org/', headers=headers)
        assert self.mw.process_response(req, res, self.spider) is res

        req2 = Request('http://scrapytest.org/sub1/')
        assert self.mw.process_request(req2, self.spider) is None
        self.assertIn('Cookie', req2.headers)

    def test_dont_merge_cookies(self):
        # merge some cookies into jar
        headers = {'Set-Cookie': 'C1=value1; path=/'}
        req = Request('http://scrapytest.org/')
        res = Response('http://scrapytest.org/', headers=headers)
        assert self.mw.process_response(req, res, self.spider) is res

        # test Cookie header is not seted to request
        req = Request('http://scrapytest.org/dontmerge', meta={'dont_merge_cookies': 1})
        assert self.mw.process_request(req, self.spider) is None
        assert 'Cookie' not in req.headers

        # check that returned cookies are not merged back to jar
        res = Response('http://scrapytest.org/dontmerge', headers={'Set-Cookie': 'dont=mergeme; path=/'})
        assert self.mw.process_response(req, res, self.spider) is res

        # check that cookies are merged back
        req = Request('http://scrapytest.org/mergeme')
        assert self.mw.process_request(req, self.spider) is None
        self.assertEqual(req.headers.get('Cookie'), b'C1=value1')

        # check that cookies are merged when dont_merge_cookies is passed as 0
        req = Request('http://scrapytest.org/mergeme', meta={'dont_merge_cookies': 0})
        assert self.mw.process_request(req, self.spider) is None
        self.assertEqual(req.headers.get('Cookie'), b'C1=value1')

    def test_complex_cookies(self):
        # merge some cookies into jar
        cookies = [{'name': 'C1', 'value': 'value1', 'path': '/foo', 'domain': 'scrapytest.org'},
                {'name': 'C2', 'value': 'value2', 'path': '/bar', 'domain': 'scrapytest.org'},
                {'name': 'C3', 'value': 'value3', 'path': '/foo', 'domain': 'scrapytest.org'},
                {'name': 'C4', 'value': 'value4', 'path': '/foo', 'domain': 'scrapy.org'}]


        req = Request('http://scrapytest.org/', cookies=cookies)
        self.mw.process_request(req, self.spider)

        # embed C1 and C3 for scrapytest.org/foo
        req = Request('http://scrapytest.org/foo')
        self.mw.process_request(req, self.spider)
        assert req.headers.get('Cookie') in (b'C1=value1; C3=value3', b'C3=value3; C1=value1')

        # embed C2 for scrapytest.org/bar
        req = Request('http://scrapytest.org/bar')
        self.mw.process_request(req, self.spider)
        self.assertEqual(req.headers.get('Cookie'), b'C2=value2')

        # embed nothing for scrapytest.org/baz
        req = Request('http://scrapytest.org/baz')
        self.mw.process_request(req, self.spider)
        assert 'Cookie' not in req.headers

    def test_merge_request_cookies(self):
        req = Request('http://scrapytest.org/', cookies={'galleta': 'salada'})
        assert self.mw.process_request(req, self.spider) is None
        self.assertEqual(req.headers.get('Cookie'), b'galleta=salada')

        headers = {'Set-Cookie': 'C1=value1; path=/'}
        res = Response('http://scrapytest.org/', headers=headers)
        assert self.mw.process_response(req, res, self.spider) is res

        req2 = Request('http://scrapytest.org/sub1/')
        assert self.mw.process_request(req2, self.spider) is None

        self.assertCookieValEqual(req2.headers.get('Cookie'), b"C1=value1; galleta=salada")

    def test_cookiejar_key(self):
        req = Request('http://scrapytest.org/', cookies={'galleta': 'salada'}, meta={'cookiejar': "store1"})
        assert self.mw.process_request(req, self.spider) is None
        self.assertEqual(req.headers.get('Cookie'), b'galleta=salada')

        headers = {'Set-Cookie': 'C1=value1; path=/'}
        res = Response('http://scrapytest.org/', headers=headers, request=req)
        assert self.mw.process_response(req, res, self.spider) is res

        req2 = Request('http://scrapytest.org/', meta=res.meta)
        assert self.mw.process_request(req2, self.spider) is None
        self.assertCookieValEqual(req2.headers.get('Cookie'), b'C1=value1; galleta=salada')

        req3 = Request('http://scrapytest.org/', cookies={'galleta': 'dulce'}, meta={'cookiejar': "store2"})
        assert self.mw.process_request(req3, self.spider) is None
        self.assertEqual(req3.headers.get('Cookie'), b'galleta=dulce')

        headers = {'Set-Cookie': 'C2=value2; path=/'}
        res2 = Response('http://scrapytest.org/', headers=headers, request=req3)
        assert self.mw.process_response(req3, res2, self.spider) is res2

        req4 = Request('http://scrapytest.org/', meta=res2.meta)
        assert self.mw.process_request(req4, self.spider) is None
        self.assertCookieValEqual(req4.headers.get('Cookie'), b'C2=value2; galleta=dulce')

        #cookies from hosts with port
        req5_1 = Request('http://scrapytest.org:1104/')
        assert self.mw.process_request(req5_1, self.spider) is None

        headers = {'Set-Cookie': 'C1=value1; path=/'}
        res5_1 = Response('http://scrapytest.org:1104/', headers=headers, request=req5_1)
        assert self.mw.process_response(req5_1, res5_1, self.spider) is res5_1

        req5_2 = Request('http://scrapytest.org:1104/some-redirected-path')
        assert self.mw.process_request(req5_2, self.spider) is None
        self.assertEqual(req5_2.headers.get('Cookie'), b'C1=value1')

        req5_3 = Request('http://scrapytest.org/some-redirected-path')
        assert self.mw.process_request(req5_3, self.spider) is None
        self.assertEqual(req5_3.headers.get('Cookie'), b'C1=value1')

        #skip cookie retrieval for not http request
        req6 = Request('file:///scrapy/sometempfile')
        assert self.mw.process_request(req6, self.spider) is None
        self.assertEqual(req6.headers.get('Cookie'), None)

    def test_local_domain(self):
        request = Request("http://example-host/", cookies={'currencyCookie': 'USD'})
        assert self.mw.process_request(request, self.spider) is None
        self.assertIn('Cookie', request.headers)
        self.assertEqual(b'currencyCookie=USD', request.headers['Cookie'])

class TestItem(Item):
    name = Field()
    age = Field()


class TestRefererMiddleware(TestCase):

    req_meta = {}
    resp_headers = {}
    settings = {}
    scenarii = [
        ('http://scrapytest.org', 'http://scrapytest.org/',  b'http://scrapytest.org'),
    ]

    def setUp(self):
        self.spider = Spider('foo')
        settings = Settings(self.settings)
        self.mw = RefererMiddleware(settings)

    def get_request(self, target):
        return Request(target, meta=self.req_meta)

    def get_response(self, origin):
        return Response(origin, headers=self.resp_headers)

    def test(self):

        for origin, target, referrer in self.scenarii:
            response = self.get_response(origin)
            request = self.get_request(target)
            out = list(self.mw.process_spider_output(response, [request], self.spider))
            self.assertEqual(out[0].headers.get('Referer'), referrer)


class ChunkSize1PickleFifoDiskQueueTest(PickleFifoDiskQueueTest):
    chunksize = 1
