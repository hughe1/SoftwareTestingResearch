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


class LogformatterSubclassTest(LoggingContribTest):
    def setUp(self):
        self.formatter = LogFormatterSubclass()
        self.spider = Spider('default')

    def test_flags_in_request(self):
        pass


if __name__ == "__main__":
    unittest.main()

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


class NestedItemLoader(ItemLoader):
    default_item_class = TestNestedItem


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

class LxmlLinkExtractorTestCase(Base.LinkExtractorTestCase):
    extractor_cls = LxmlLinkExtractor

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

    @pytest.mark.xfail
    def test_restrict_xpaths_with_html_entities(self):
        super(LxmlLinkExtractorTestCase, self).test_restrict_xpaths_with_html_entities()


class MySpider1(MyBaseSpider):
    name = 'myspider1'

class BenchCommandTest(CommandTest):

    def test_run(self):
        p = self.proc('bench', '-s', 'LOGSTATS_INTERVAL=0.001',
                '-s', 'CLOSESPIDER_TIMEOUT=0.01')
        log = to_native_str(p.stderr.read())
        self.assertIn('INFO: Crawled', log)
        self.assertNotIn('Unhandled Error', log)

class PickleItemExporterTest(BaseItemExporterTest):

    def _get_exporter(self, **kwargs):
        return PickleItemExporter(self.output, **kwargs)

    def _check_output(self):
        self._assert_expected_item(pickle.loads(self.output.getvalue()))

    def test_export_multiple_items(self):
        i1 = TestItem(name='hello', age='world')
        i2 = TestItem(name='bye', age='world')
        f = BytesIO()
        ie = PickleItemExporter(f)
        ie.start_exporting()
        ie.export_item(i1)
        ie.export_item(i2)
        ie.finish_exporting()
        f.seek(0)
        self.assertEqual(pickle.load(f), i1)
        self.assertEqual(pickle.load(f), i2)

    def test_nonstring_types_item(self):
        item = self._get_nonstring_types_item()
        fp = BytesIO()
        ie = PickleItemExporter(fp)
        ie.start_exporting()
        ie.export_item(item)
        ie.finish_exporting()
        self.assertEqual(pickle.loads(fp.getvalue()), item)


class FilesPipelineTestCase(unittest.TestCase):

    def setUp(self):
        self.tempdir = mkdtemp()
        self.pipeline = FilesPipeline.from_settings(Settings({'FILES_STORE': self.tempdir}))
        self.pipeline.download_func = _mocked_download_func
        self.pipeline.open_spider(None)

    def tearDown(self):
        rmtree(self.tempdir)

    def test_file_path(self):
        file_path = self.pipeline.file_path
        self.assertEqual(file_path(Request("https://dev.mydeco.com/mydeco.pdf")),
                         'full/c9b564df929f4bc635bdd19fde4f3d4847c757c5.pdf')
        self.assertEqual(file_path(Request("http://www.maddiebrown.co.uk///catalogue-items//image_54642_12175_95307.txt")),
                         'full/4ce274dd83db0368bafd7e406f382ae088e39219.txt')
        self.assertEqual(file_path(Request("https://dev.mydeco.com/two/dirs/with%20spaces%2Bsigns.doc")),
                         'full/94ccc495a17b9ac5d40e3eabf3afcb8c2c9b9e1a.doc')
        self.assertEqual(file_path(Request("http://www.dfsonline.co.uk/get_prod_image.php?img=status_0907_mdm.jpg")),
                         'full/4507be485f38b0da8a0be9eb2e1dfab8a19223f2.jpg')
        self.assertEqual(file_path(Request("http://www.dorma.co.uk/images/product_details/2532/")),
                         'full/97ee6f8a46cbbb418ea91502fd24176865cf39b2')
        self.assertEqual(file_path(Request("http://www.dorma.co.uk/images/product_details/2532")),
                         'full/244e0dd7d96a3b7b01f54eded250c9e272577aa1')
        self.assertEqual(file_path(Request("http://www.dorma.co.uk/images/product_details/2532"),
                                   response=Response("http://www.dorma.co.uk/images/product_details/2532"),
                                   info=object()),
                         'full/244e0dd7d96a3b7b01f54eded250c9e272577aa1')

    def test_fs_store(self):
        assert isinstance(self.pipeline.store, FSFilesStore)
        self.assertEqual(self.pipeline.store.basedir, self.tempdir)

        path = 'some/image/key.jpg'
        fullpath = os.path.join(self.tempdir, 'some', 'image', 'key.jpg')
        self.assertEqual(self.pipeline.store._get_filesystem_path(path), fullpath)

    @defer.inlineCallbacks
    def test_file_not_expired(self):
        item_url = "http://example.com/file.pdf"
        item = _create_item_with_files(item_url)
        patchers = [
            mock.patch.object(FilesPipeline, 'inc_stats', return_value=True),
            mock.patch.object(FSFilesStore, 'stat_file', return_value={
                'checksum': 'abc', 'last_modified': time.time()}),
            mock.patch.object(FilesPipeline, 'get_media_requests',
                              return_value=[_prepare_request_object(item_url)])
        ]
        for p in patchers:
            p.start()

        result = yield self.pipeline.process_item(item, None)
        self.assertEqual(result['files'][0]['checksum'], 'abc')

        for p in patchers:
            p.stop()

    @defer.inlineCallbacks
    def test_file_expired(self):
        item_url = "http://example.com/file2.pdf"
        item = _create_item_with_files(item_url)
        patchers = [
            mock.patch.object(FSFilesStore, 'stat_file', return_value={
                'checksum': 'abc',
                'last_modified': time.time() - (self.pipeline.expires * 60 * 60 * 24 * 2)}),
            mock.patch.object(FilesPipeline, 'get_media_requests',
                              return_value=[_prepare_request_object(item_url)]),
            mock.patch.object(FilesPipeline, 'inc_stats', return_value=True)
        ]
        for p in patchers:
            p.start()

        result = yield self.pipeline.process_item(item, None)
        self.assertNotEqual(result['files'][0]['checksum'], 'abc')

        for p in patchers:
            p.stop()


class TestReferrerOnRedirectStrictOriginWhenCrossOrigin(TestReferrerOnRedirect):
    """
    Strict Origin When Cross-Origin policy sends the full URL as "Referer",
    unless the target's origin is different (different domain, different protocol)
    in which case only the origin is sent...
    Unless there's also a downgrade in security and then the "Referer" header
    is not sent.

    Redirections to a different origin should strip the "Referer" to the parent origin,
    and from https:// to http:// will remove the "Referer" header.
    """
    settings = {'REFERRER_POLICY': POLICY_STRICT_ORIGIN_WHEN_CROSS_ORIGIN}
    scenarii = [
        (   'http://scrapytest.org/101',      # origin
            'http://scrapytest.org/102',      # target + redirection
            (
                # redirections: code, URL
                (301, 'http://scrapytest.org/103'),
                (301, 'http://scrapytest.org/104'),
            ),
            b'http://scrapytest.org/101', # expected initial referer
            b'http://scrapytest.org/101', # expected referer for the redirection request
        ),
        (   'https://scrapytest.org/201',
            'https://scrapytest.org/202',
            (
                # redirecting to non-secure URL: do not send the "Referer" header
                (301, 'http://scrapytest.org/203'),
            ),
            b'https://scrapytest.org/201',
            None,
        ),
        (   'https://scrapytest.org/301',
            'https://scrapytest.org/302',
            (
                # redirecting to non-secure URL (different domain): send origin
                (301, 'http://example.com/303'),
            ),
            b'https://scrapytest.org/301',
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
                # all different domains: send origin
                (301, 'https://google.com/503'),
                (301, 'https://facebook.com/504'),
            ),
            b'https://scrapy.org/',
            b'https://scrapy.org/',
        ),
        (   'https://scrapytest.org/601',
            'http://scrapytest.org/602',                # TLS to non-TLS: do not send "Referer"
            (
                (301, 'https://scrapytest.org/603'),    # TLS URL again: (still) send nothing
            ),
            None,
            None,
        ),
    ]

class SelectortemLoaderTest(unittest.TestCase):
    response = HtmlResponse(url="", encoding='utf-8', body=b"""
    <html>
    <body>
    <div id="id">marta</div>
    <p>paragraph</p>
    <a href="http://www.scrapy.org">homepage</a>
    <img src="/images/logo.png" width="244" height="65" alt="Scrapy">
    </body>
    </html>
    """)

    def test_constructor(self):
        l = TestItemLoader()
        self.assertEqual(l.selector, None)

    def test_constructor_errors(self):
        l = TestItemLoader()
        self.assertRaises(RuntimeError, l.add_xpath, 'url', '//a/@href')
        self.assertRaises(RuntimeError, l.replace_xpath, 'url', '//a/@href')
        self.assertRaises(RuntimeError, l.get_xpath, '//a/@href')
        self.assertRaises(RuntimeError, l.add_css, 'name', '#name::text')
        self.assertRaises(RuntimeError, l.replace_css, 'name', '#name::text')
        self.assertRaises(RuntimeError, l.get_css, '#name::text')

    def test_constructor_with_selector(self):
        sel = Selector(text=u"<html><body><div>marta</div></body></html>")
        l = TestItemLoader(selector=sel)
        self.assertIs(l.selector, sel)

        l.add_xpath('name', '//div/text()')
        self.assertEqual(l.get_output_value('name'), [u'Marta'])

    def test_constructor_with_selector_css(self):
        sel = Selector(text=u"<html><body><div>marta</div></body></html>")
        l = TestItemLoader(selector=sel)
        self.assertIs(l.selector, sel)

        l.add_css('name', 'div::text')
        self.assertEqual(l.get_output_value('name'), [u'Marta'])

    def test_constructor_with_response(self):
        l = TestItemLoader(response=self.response)
        self.assertTrue(l.selector)

        l.add_xpath('name', '//div/text()')
        self.assertEqual(l.get_output_value('name'), [u'Marta'])

    def test_constructor_with_response_css(self):
        l = TestItemLoader(response=self.response)
        self.assertTrue(l.selector)

        l.add_css('name', 'div::text')
        self.assertEqual(l.get_output_value('name'), [u'Marta'])

        l.add_css('url', 'a::attr(href)')
        self.assertEqual(l.get_output_value('url'), [u'http://www.scrapy.org'])

        # combining/accumulating CSS selectors and XPath expressions
        l.add_xpath('name', '//div/text()')
        self.assertEqual(l.get_output_value('name'), [u'Marta', u'Marta'])

        l.add_xpath('url', '//img/@src')
        self.assertEqual(l.get_output_value('url'), [u'http://www.scrapy.org', u'/images/logo.png'])

    def test_add_xpath_re(self):
        l = TestItemLoader(response=self.response)
        l.add_xpath('name', '//div/text()', re='ma')
        self.assertEqual(l.get_output_value('name'), [u'Ma'])

    def test_replace_xpath(self):
        l = TestItemLoader(response=self.response)
        self.assertTrue(l.selector)
        l.add_xpath('name', '//div/text()')
        self.assertEqual(l.get_output_value('name'), [u'Marta'])
        l.replace_xpath('name', '//p/text()')
        self.assertEqual(l.get_output_value('name'), [u'Paragraph'])

        l.replace_xpath('name', ['//p/text()', '//div/text()'])
        self.assertEqual(l.get_output_value('name'), [u'Paragraph', 'Marta'])

    def test_get_xpath(self):
        l = TestItemLoader(response=self.response)
        self.assertEqual(l.get_xpath('//p/text()'), [u'paragraph'])
        self.assertEqual(l.get_xpath('//p/text()', TakeFirst()), u'paragraph')
        self.assertEqual(l.get_xpath('//p/text()', TakeFirst(), re='pa'), u'pa')

        self.assertEqual(l.get_xpath(['//p/text()', '//div/text()']), [u'paragraph', 'marta'])

    def test_replace_xpath_multi_fields(self):
        l = TestItemLoader(response=self.response)
        l.add_xpath(None, '//div/text()', TakeFirst(), lambda x: {'name': x})
        self.assertEqual(l.get_output_value('name'), [u'Marta'])
        l.replace_xpath(None, '//p/text()', TakeFirst(), lambda x: {'name': x})
        self.assertEqual(l.get_output_value('name'), [u'Paragraph'])

    def test_replace_xpath_re(self):
        l = TestItemLoader(response=self.response)
        self.assertTrue(l.selector)
        l.add_xpath('name', '//div/text()')
        self.assertEqual(l.get_output_value('name'), [u'Marta'])
        l.replace_xpath('name', '//div/text()', re='ma')
        self.assertEqual(l.get_output_value('name'), [u'Ma'])

    def test_add_css_re(self):
        l = TestItemLoader(response=self.response)
        l.add_css('name', 'div::text', re='ma')
        self.assertEqual(l.get_output_value('name'), [u'Ma'])

        l.add_css('url', 'a::attr(href)', re='http://(.+)')
        self.assertEqual(l.get_output_value('url'), [u'www.scrapy.org'])

    def test_replace_css(self):
        l = TestItemLoader(response=self.response)
        self.assertTrue(l.selector)
        l.add_css('name', 'div::text')
        self.assertEqual(l.get_output_value('name'), [u'Marta'])
        l.replace_css('name', 'p::text')
        self.assertEqual(l.get_output_value('name'), [u'Paragraph'])

        l.replace_css('name', ['p::text', 'div::text'])
        self.assertEqual(l.get_output_value('name'), [u'Paragraph', 'Marta'])

        l.add_css('url', 'a::attr(href)', re='http://(.+)')
        self.assertEqual(l.get_output_value('url'), [u'www.scrapy.org'])
        l.replace_css('url', 'img::attr(src)')
        self.assertEqual(l.get_output_value('url'), [u'/images/logo.png'])

    def test_get_css(self):
        l = TestItemLoader(response=self.response)
        self.assertEqual(l.get_css('p::text'), [u'paragraph'])
        self.assertEqual(l.get_css('p::text', TakeFirst()), u'paragraph')
        self.assertEqual(l.get_css('p::text', TakeFirst(), re='pa'), u'pa')

        self.assertEqual(l.get_css(['p::text', 'div::text']), [u'paragraph', 'marta'])
        self.assertEqual(l.get_css(['a::attr(href)', 'img::attr(src)']),
            [u'http://www.scrapy.org', u'/images/logo.png'])

    def test_replace_css_multi_fields(self):
        l = TestItemLoader(response=self.response)
        l.add_css(None, 'div::text', TakeFirst(), lambda x: {'name': x})
        self.assertEqual(l.get_output_value('name'), [u'Marta'])
        l.replace_css(None, 'p::text', TakeFirst(), lambda x: {'name': x})
        self.assertEqual(l.get_output_value('name'), [u'Paragraph'])

        l.add_css(None, 'a::attr(href)', TakeFirst(), lambda x: {'url': x})
        self.assertEqual(l.get_output_value('url'), [u'http://www.scrapy.org'])
        l.replace_css(None, 'img::attr(src)', TakeFirst(), lambda x: {'url': x})
        self.assertEqual(l.get_output_value('url'), [u'/images/logo.png'])

    def test_replace_css_re(self):
        l = TestItemLoader(response=self.response)
        self.assertTrue(l.selector)
        l.add_css('url', 'a::attr(href)')
        self.assertEqual(l.get_output_value('url'), [u'http://www.scrapy.org'])
        l.replace_css('url', 'a::attr(href)', re='http://www\.(.+)')
        self.assertEqual(l.get_output_value('url'), [u'scrapy.org'])


class UtilsPythonTestCase(unittest.TestCase):

    def test_equal_attributes(self):
        class Obj:
            pass

        a = Obj()
        b = Obj()
        # no attributes given return False
        self.assertFalse(equal_attributes(a, b, []))
        # not existent attributes
        self.assertFalse(equal_attributes(a, b, ['x', 'y']))

        a.x = 1
        b.x = 1
        # equal attribute
        self.assertTrue(equal_attributes(a, b, ['x']))

        b.y = 2
        # obj1 has no attribute y
        self.assertFalse(equal_attributes(a, b, ['x', 'y']))

        a.y = 2
        # equal attributes
        self.assertTrue(equal_attributes(a, b, ['x', 'y']))

        a.y = 1
        # differente attributes
        self.assertFalse(equal_attributes(a, b, ['x', 'y']))

        # test callable
        a.meta = {}
        b.meta = {}
        self.assertTrue(equal_attributes(a, b, ['meta']))

        # compare ['meta']['a']
        a.meta['z'] = 1
        b.meta['z'] = 1

        get_z = operator.itemgetter('z')
        get_meta = operator.attrgetter('meta')
        compare_z = lambda obj: get_z(get_meta(obj))

        self.assertTrue(equal_attributes(a, b, [compare_z, 'x']))
        # fail z equality
        a.meta['z'] = 2
        self.assertFalse(equal_attributes(a, b, [compare_z, 'x']))

    def test_weakkeycache(self):
        class _Weakme(object): pass
        _values = count()
        wk = WeakKeyCache(lambda k: next(_values))
        k = _Weakme()
        v = wk[k]
        self.assertEqual(v, wk[k])
        self.assertNotEqual(v, wk[_Weakme()])
        self.assertEqual(v, wk[k])
        del k
        for _ in range(100):
            if wk._weakdict:
                gc.collect()
        self.assertFalse(len(wk._weakdict))

    @unittest.skipUnless(six.PY2, "deprecated function")
    def test_stringify_dict(self):
        d = {'a': 123, u'b': b'c', u'd': u'e', object(): u'e'}
        d2 = stringify_dict(d, keys_only=False)
        self.assertEqual(d, d2)
        self.assertIsNot(d, d2)  # shouldn't modify in place
        self.assertFalse(any(isinstance(x, six.text_type) for x in d2.keys()))
        self.assertFalse(any(isinstance(x, six.text_type) for x in d2.values()))

    @unittest.skipUnless(six.PY2, "deprecated function")
    def test_stringify_dict_tuples(self):
        tuples = [('a', 123), (u'b', 'c'), (u'd', u'e'), (object(), u'e')]
        d = dict(tuples)
        d2 = stringify_dict(tuples, keys_only=False)
        self.assertEqual(d, d2)
        self.assertIsNot(d, d2)  # shouldn't modify in place
        self.assertFalse(any(isinstance(x, six.text_type) for x in d2.keys()), d2.keys())
        self.assertFalse(any(isinstance(x, six.text_type) for x in d2.values()))

    @unittest.skipUnless(six.PY2, "deprecated function")
    def test_stringify_dict_keys_only(self):
        d = {'a': 123, u'b': 'c', u'd': u'e', object(): u'e'}
        d2 = stringify_dict(d)
        self.assertEqual(d, d2)
        self.assertIsNot(d, d2)  # shouldn't modify in place
        self.assertFalse(any(isinstance(x, six.text_type) for x in d2.keys()))

    def test_get_func_args(self):
        def f1(a, b, c):
            pass

        def f2(a, b=None, c=None):
            pass

        class A(object):
            def __init__(self, a, b, c):
                pass

            def method(self, a, b, c):
                pass

        class Callable(object):

            def __call__(self, a, b, c):
                pass

        a = A(1, 2, 3)
        cal = Callable()
        partial_f1 = functools.partial(f1, None)
        partial_f2 = functools.partial(f1, b=None)
        partial_f3 = functools.partial(partial_f2, None)

        self.assertEqual(get_func_args(f1), ['a', 'b', 'c'])
        self.assertEqual(get_func_args(f2), ['a', 'b', 'c'])
        self.assertEqual(get_func_args(A), ['a', 'b', 'c'])
        self.assertEqual(get_func_args(a.method), ['a', 'b', 'c'])
        self.assertEqual(get_func_args(partial_f1), ['b', 'c'])
        self.assertEqual(get_func_args(partial_f2), ['a', 'c'])
        self.assertEqual(get_func_args(partial_f3), ['c'])
        self.assertEqual(get_func_args(cal), ['a', 'b', 'c'])
        self.assertEqual(get_func_args(object), [])

        if platform.python_implementation() == 'CPython':
            # TODO: how do we fix this to return the actual argument names?
            self.assertEqual(get_func_args(six.text_type.split), [])
            self.assertEqual(get_func_args(" ".join), [])
            self.assertEqual(get_func_args(operator.itemgetter(2)), [])
        else:
            self.assertEqual(get_func_args(six.text_type.split), ['sep', 'maxsplit'])
            self.assertEqual(get_func_args(" ".join), ['list'])
            self.assertEqual(get_func_args(operator.itemgetter(2)), ['obj'])


    def test_without_none_values(self):
        self.assertEqual(without_none_values([1, None, 3, 4]), [1, 3, 4])
        self.assertEqual(without_none_values((1, None, 3, 4)), (1, 3, 4))
        self.assertEqual(
            without_none_values({'one': 1, 'none': None, 'three': 3, 'four': 4}),
            {'one': 1, 'three': 3, 'four': 4})

if __name__ == "__main__":
    unittest.main()

class BaseResponseTest(unittest.TestCase):

    response_class = Response

    def test_init(self):
        # Response requires url in the consturctor
        self.assertRaises(Exception, self.response_class)
        self.assertTrue(isinstance(self.response_class('http://example.com/'), self.response_class))
        if not six.PY2:
            self.assertRaises(TypeError, self.response_class, b"http://example.com")
        # body can be str or None
        self.assertTrue(isinstance(self.response_class('http://example.com/', body=b''), self.response_class))
        self.assertTrue(isinstance(self.response_class('http://example.com/', body=b'body'), self.response_class))
        # test presence of all optional parameters
        self.assertTrue(isinstance(self.response_class('http://example.com/', body=b'', headers={}, status=200), self.response_class))

        r = self.response_class("http://www.example.com")
        assert isinstance(r.url, str)
        self.assertEqual(r.url, "http://www.example.com")
        self.assertEqual(r.status, 200)

        assert isinstance(r.headers, Headers)
        self.assertEqual(r.headers, {})

        headers = {"foo": "bar"}
        body = b"a body"
        r = self.response_class("http://www.example.com", headers=headers, body=body)

        assert r.headers is not headers
        self.assertEqual(r.headers[b"foo"], b"bar")

        r = self.response_class("http://www.example.com", status=301)
        self.assertEqual(r.status, 301)
        r = self.response_class("http://www.example.com", status='301')
        self.assertEqual(r.status, 301)
        self.assertRaises(ValueError, self.response_class, "http://example.com", status='lala200')

    def test_copy(self):
        """Test Response copy"""

        r1 = self.response_class("http://www.example.com", body=b"Some body")
        r1.flags.append('cached')
        r2 = r1.copy()

        self.assertEqual(r1.status, r2.status)
        self.assertEqual(r1.body, r2.body)

        # make sure flags list is shallow copied
        assert r1.flags is not r2.flags, "flags must be a shallow copy, not identical"
        self.assertEqual(r1.flags, r2.flags)

        # make sure headers attribute is shallow copied
        assert r1.headers is not r2.headers, "headers must be a shallow copy, not identical"
        self.assertEqual(r1.headers, r2.headers)

    def test_copy_meta(self):
        req = Request("http://www.example.com")
        req.meta['foo'] = 'bar'
        r1 = self.response_class("http://www.example.com", body=b"Some body", request=req)
        assert r1.meta is req.meta

    def test_copy_inherited_classes(self):
        """Test Response children copies preserve their class"""

        class CustomResponse(self.response_class):
            pass

        r1 = CustomResponse('http://www.example.com')
        r2 = r1.copy()

        assert type(r2) is CustomResponse

    def test_replace(self):
        """Test Response.replace() method"""
        hdrs = Headers({"key": "value"})
        r1 = self.response_class("http://www.example.com")
        r2 = r1.replace(status=301, body=b"New body", headers=hdrs)
        assert r1.body == b''
        self.assertEqual(r1.url, r2.url)
        self.assertEqual((r1.status, r2.status), (200, 301))
        self.assertEqual((r1.body, r2.body), (b'', b"New body"))
        self.assertEqual((r1.headers, r2.headers), ({}, hdrs))

        # Empty attributes (which may fail if not compared properly)
        r3 = self.response_class("http://www.example.com", flags=['cached'])
        r4 = r3.replace(body=b'', flags=[])
        self.assertEqual(r4.body, b'')
        self.assertEqual(r4.flags, [])

    def _assert_response_values(self, response, encoding, body):
        if isinstance(body, six.text_type):
            body_unicode = body
            body_bytes = body.encode(encoding)
        else:
            body_unicode = body.decode(encoding)
            body_bytes = body

        assert isinstance(response.body, bytes)
        assert isinstance(response.text, six.text_type)
        self._assert_response_encoding(response, encoding)
        self.assertEqual(response.body, body_bytes)
        self.assertEqual(response.body_as_unicode(), body_unicode)
        self.assertEqual(response.text, body_unicode)

    def _assert_response_encoding(self, response, encoding):
        self.assertEqual(response.encoding, resolve_encoding(encoding))

    def test_immutable_attributes(self):
        r = self.response_class("http://example.com")
        self.assertRaises(AttributeError, setattr, r, 'url', 'http://example2.com')
        self.assertRaises(AttributeError, setattr, r, 'body', 'xxx')

    def test_urljoin(self):
        """Test urljoin shortcut (only for existence, since behavior equals urljoin)"""
        joined = self.response_class('http://www.example.com').urljoin('/test')
        absolute = 'http://www.example.com/test'
        self.assertEqual(joined, absolute)

    def test_shortcut_attributes(self):
        r = self.response_class("http://example.com", body=b'hello')
        if self.response_class == Response:
            msg = "Response content isn't text"
            self.assertRaisesRegexp(AttributeError, msg, getattr, r, 'text')
            self.assertRaisesRegexp(NotSupported, msg, r.css, 'body')
            self.assertRaisesRegexp(NotSupported, msg, r.xpath, '//body')
        else:
            r.text
            r.css('body')
            r.xpath('//body')

    def test_follow_url_absolute(self):
        self._assert_followed_url('http://foo.example.com',
                                  'http://foo.example.com')

    def test_follow_url_relative(self):
        self._assert_followed_url('foo',
                                  'http://example.com/foo')

    def test_follow_link(self):
        self._assert_followed_url(Link('http://example.com/foo'),
                                  'http://example.com/foo')

    def test_follow_whitespace_url(self):
        self._assert_followed_url('foo ',
                                  'http://example.com/foo%20')

    def test_follow_whitespace_link(self):
        self._assert_followed_url(Link('http://example.com/foo '),
                                  'http://example.com/foo%20')
    def _assert_followed_url(self, follow_obj, target_url, response=None):
        if response is None:
            response = self._links_response()
        req = response.follow(follow_obj)
        self.assertEqual(req.url, target_url)
        return req

    def _links_response(self):
        body = get_testdata('link_extractor', 'sgml_linkextractor.html')
        resp = self.response_class('http://example.com/index', body=body)
        return resp


class Https11TestCase(Http11TestCase):
    scheme = 'https'


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


class MustbeDeferredTest(unittest.TestCase):
    def test_success_function(self):
        steps = []
        def _append(v):
            steps.append(v)
            return steps

        dfd = mustbe_deferred(_append, 1)
        dfd.addCallback(self.assertEqual, [1, 2]) # it is [1] with maybeDeferred
        steps.append(2) # add another value, that should be catched by assertEqual
        return dfd

    def test_unfired_deferred(self):
        steps = []
        def _append(v):
            steps.append(v)
            dfd = defer.Deferred()
            reactor.callLater(0, dfd.callback, steps)
            return dfd

        dfd = mustbe_deferred(_append, 1)
        dfd.addCallback(self.assertEqual, [1, 2]) # it is [1] with maybeDeferred
        steps.append(2) # add another value, that should be catched by assertEqual
        return dfd

def cb1(value, arg1, arg2):
    return "(cb1 %s %s %s)" % (value, arg1, arg2)
def cb2(value, arg1, arg2):
    return defer.succeed("(cb2 %s %s %s)" % (value, arg1, arg2))
def cb3(value, arg1, arg2):
    return "(cb3 %s %s %s)" % (value, arg1, arg2)
def cb_fail(value, arg1, arg2):
    return Failure(TypeError())
def eb1(failure, arg1, arg2):
    return "(eb1 %s %s %s)" % (failure.value.__class__.__name__, arg1, arg2)


class MockedMediaPipeline(MediaPipeline):

    def __init__(self, *args, **kwargs):
        super(MockedMediaPipeline, self).__init__(*args, **kwargs)
        self._mockcalled = []

    def download(self, request, info):
        self._mockcalled.append('download')
        return super(MockedMediaPipeline, self).download(request, info)

    def media_to_download(self, request, info):
        self._mockcalled.append('media_to_download')
        if 'result' in request.meta:
            return request.meta.get('result')
        return super(MockedMediaPipeline, self).media_to_download(request, info)

    def get_media_requests(self, item, info):
        self._mockcalled.append('get_media_requests')
        return item.get('requests')

    def media_downloaded(self, response, request, info):
        self._mockcalled.append('media_downloaded')
        return super(MockedMediaPipeline, self).media_downloaded(response, request, info)

    def media_failed(self, failure, request, info):
        self._mockcalled.append('media_failed')
        return super(MockedMediaPipeline, self).media_failed(failure, request, info)

    def item_completed(self, results, item, info):
        self._mockcalled.append('item_completed')
        item = super(MockedMediaPipeline, self).item_completed(results, item, info)
        item['results'] = results
        return item


class _HttpErrorSpider(Spider):
    name = 'httperror'
    start_urls = [
        "http://localhost:8998/status?n=200",
        "http://localhost:8998/status?n=404",
        "http://localhost:8998/status?n=402",
        "http://localhost:8998/status?n=500",
    ]
    bypass_status_codes = set()

    def __init__(self, *args, **kwargs):
        super(_HttpErrorSpider, self).__init__(*args, **kwargs)
        self.failed = set()
        self.skipped = set()
        self.parsed = set()

    def start_requests(self):
        for url in self.start_urls:
            yield Request(url, self.parse, errback=self.on_error)

    def parse(self, response):
        self.parsed.add(response.url[-3:])

    def on_error(self, failure):
        if isinstance(failure.value, HttpError):
            response = failure.value.response
            if response.status in self.bypass_status_codes:
                self.skipped.add(response.url[-3:])
                return self.parse(response)

        # it assumes there is a response attached to failure
        self.failed.add(failure.value.response.url[-3:])
        return failure


def _responses(request, status_codes):
    responses = []
    for code in status_codes:
        response = Response(request.url, status=code)
        response.request = request
        responses.append(response)
    return responses


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


class MustbeDeferredTest(unittest.TestCase):
    def test_success_function(self):
        steps = []
        def _append(v):
            steps.append(v)
            return steps

        dfd = mustbe_deferred(_append, 1)
        dfd.addCallback(self.assertEqual, [1, 2]) # it is [1] with maybeDeferred
        steps.append(2) # add another value, that should be catched by assertEqual
        return dfd

    def test_unfired_deferred(self):
        steps = []
        def _append(v):
            steps.append(v)
            dfd = defer.Deferred()
            reactor.callLater(0, dfd.callback, steps)
            return dfd

        dfd = mustbe_deferred(_append, 1)
        dfd.addCallback(self.assertEqual, [1, 2]) # it is [1] with maybeDeferred
        steps.append(2) # add another value, that should be catched by assertEqual
        return dfd

def cb1(value, arg1, arg2):
    return "(cb1 %s %s %s)" % (value, arg1, arg2)
def cb2(value, arg1, arg2):
    return defer.succeed("(cb2 %s %s %s)" % (value, arg1, arg2))
def cb3(value, arg1, arg2):
    return "(cb3 %s %s %s)" % (value, arg1, arg2)
def cb_fail(value, arg1, arg2):
    return Failure(TypeError())
def eb1(failure, arg1, arg2):
    return "(eb1 %s %s %s)" % (failure.value.__class__.__name__, arg1, arg2)


class FeedExportTest(unittest.TestCase):

    class MyItem(scrapy.Item):
        foo = scrapy.Field()
        egg = scrapy.Field()
        baz = scrapy.Field()

    @defer.inlineCallbacks
    def run_and_export(self, spider_cls, settings=None):
        """ Run spider with specified settings; return exported data. """
        tmpdir = tempfile.mkdtemp()
        res_name = tmpdir + '/res'
        defaults = {
            'FEED_URI': 'file://' + res_name,
            'FEED_FORMAT': 'csv',
        }
        defaults.update(settings or {})
        try:
            with MockServer() as s:
                runner = CrawlerRunner(Settings(defaults))
                yield runner.crawl(spider_cls)

            with open(res_name, 'rb') as f:
                defer.returnValue(f.read())

        finally:
            shutil.rmtree(tmpdir)

    @defer.inlineCallbacks
    def exported_data(self, items, settings):
        """
        Return exported data which a spider yielding ``items`` would return.
        """
        class TestSpider(scrapy.Spider):
            name = 'testspider'
            start_urls = ['http://localhost:8998/']

            def parse(self, response):
                for item in items:
                    yield item

        data = yield self.run_and_export(TestSpider, settings)
        defer.returnValue(data)

    @defer.inlineCallbacks
    def exported_no_data(self, settings):
        """
        Return exported data which a spider yielding no ``items`` would return.
        """
        class TestSpider(scrapy.Spider):
            name = 'testspider'
            start_urls = ['http://localhost:8998/']

            def parse(self, response):
                pass

        data = yield self.run_and_export(TestSpider, settings)
        defer.returnValue(data)

    @defer.inlineCallbacks
    def assertExportedCsv(self, items, header, rows, settings=None, ordered=True):
        settings = settings or {}
        settings.update({'FEED_FORMAT': 'csv'})
        data = yield self.exported_data(items, settings)

        reader = csv.DictReader(to_native_str(data).splitlines())
        got_rows = list(reader)
        if ordered:
            self.assertEqual(reader.fieldnames, header)
        else:
            self.assertEqual(set(reader.fieldnames), set(header))

        self.assertEqual(rows, got_rows)

    @defer.inlineCallbacks
    def assertExportedJsonLines(self, items, rows, settings=None):
        settings = settings or {}
        settings.update({'FEED_FORMAT': 'jl'})
        data = yield self.exported_data(items, settings)
        parsed = [json.loads(to_native_str(line)) for line in data.splitlines()]
        rows = [{k: v for k, v in row.items() if v} for row in rows]
        self.assertEqual(rows, parsed)

    @defer.inlineCallbacks
    def assertExportedXml(self, items, rows, settings=None):
        settings = settings or {}
        settings.update({'FEED_FORMAT': 'xml'})
        data = yield self.exported_data(items, settings)
        rows = [{k: v for k, v in row.items() if v} for row in rows]
        import lxml.etree
        root = lxml.etree.fromstring(data)
        got_rows = [{e.tag: e.text for e in it} for it in root.findall('item')]
        self.assertEqual(rows, got_rows)

    def _load_until_eof(self, data, load_func):
        bytes_output = BytesIO(data)
        result = []
        while True:
            try:
                result.append(load_func(bytes_output))
            except EOFError:
                break
        return result

    @defer.inlineCallbacks
    def assertExportedPickle(self, items, rows, settings=None):
        settings = settings or {}
        settings.update({'FEED_FORMAT': 'pickle'})
        data = yield self.exported_data(items, settings)
        expected = [{k: v for k, v in row.items() if v} for row in rows]
        import pickle
        result = self._load_until_eof(data, load_func=pickle.load)
        self.assertEqual(expected, result)

    @defer.inlineCallbacks
    def assertExportedMarshal(self, items, rows, settings=None):
        settings = settings or {}
        settings.update({'FEED_FORMAT': 'marshal'})
        data = yield self.exported_data(items, settings)
        expected = [{k: v for k, v in row.items() if v} for row in rows]
        import marshal
        result = self._load_until_eof(data, load_func=marshal.load)
        self.assertEqual(expected, result)

    @defer.inlineCallbacks
    def assertExported(self, items, header, rows, settings=None, ordered=True):
        yield self.assertExportedCsv(items, header, rows, settings, ordered)
        yield self.assertExportedJsonLines(items, rows, settings)
        yield self.assertExportedXml(items, rows, settings)
        yield self.assertExportedPickle(items, rows, settings)

    @defer.inlineCallbacks
    def test_export_items(self):
        # feed exporters use field names from Item
        items = [
            self.MyItem({'foo': 'bar1', 'egg': 'spam1'}),
            self.MyItem({'foo': 'bar2', 'egg': 'spam2', 'baz': 'quux2'}),
        ]
        rows = [
            {'egg': 'spam1', 'foo': 'bar1', 'baz': ''},
            {'egg': 'spam2', 'foo': 'bar2', 'baz': 'quux2'}
        ]
        header = self.MyItem.fields.keys()
        yield self.assertExported(items, header, rows, ordered=False)

    @defer.inlineCallbacks
    def test_export_no_items_not_store_empty(self):
        formats = ('json',
                   'jsonlines',
                   'xml',
                   'csv',)

        for fmt in formats:
            settings = {'FEED_FORMAT': fmt}
            data = yield self.exported_no_data(settings)
            self.assertEqual(data, b'')

    @defer.inlineCallbacks
    def test_export_no_items_store_empty(self):
        formats = (
            ('json', b'[]'),
            ('jsonlines', b''),
            ('xml', b'<?xml version="1.0" encoding="utf-8"?>\n<items></items>'),
            ('csv', b''),
        )

        for fmt, expctd in formats:
            settings = {'FEED_FORMAT': fmt, 'FEED_STORE_EMPTY': True, 'FEED_EXPORT_INDENT': None}
            data = yield self.exported_no_data(settings)
            self.assertEqual(data, expctd)

    @defer.inlineCallbacks
    def test_export_multiple_item_classes(self):

        class MyItem2(scrapy.Item):
            foo = scrapy.Field()
            hello = scrapy.Field()

        items = [
            self.MyItem({'foo': 'bar1', 'egg': 'spam1'}),
            MyItem2({'hello': 'world2', 'foo': 'bar2'}),
            self.MyItem({'foo': 'bar3', 'egg': 'spam3', 'baz': 'quux3'}),
            {'hello': 'world4', 'egg': 'spam4'},
        ]

        # by default, Scrapy uses fields of the first Item for CSV and
        # all fields for JSON Lines
        header = self.MyItem.fields.keys()
        rows_csv = [
            {'egg': 'spam1', 'foo': 'bar1', 'baz': ''},
            {'egg': '',      'foo': 'bar2', 'baz': ''},
            {'egg': 'spam3', 'foo': 'bar3', 'baz': 'quux3'},
            {'egg': 'spam4', 'foo': '',     'baz': ''},
        ]
        rows_jl = [dict(row) for row in items]
        yield self.assertExportedCsv(items, header, rows_csv, ordered=False)
        yield self.assertExportedJsonLines(items, rows_jl)

        # edge case: FEED_EXPORT_FIELDS==[] means the same as default None
        settings = {'FEED_EXPORT_FIELDS': []}
        yield self.assertExportedCsv(items, header, rows_csv, ordered=False)
        yield self.assertExportedJsonLines(items, rows_jl, settings)

        # it is possible to override fields using FEED_EXPORT_FIELDS
        header = ["foo", "baz", "hello"]
        settings = {'FEED_EXPORT_FIELDS': header}
        rows = [
            {'foo': 'bar1', 'baz': '',      'hello': ''},
            {'foo': 'bar2', 'baz': '',      'hello': 'world2'},
            {'foo': 'bar3', 'baz': 'quux3', 'hello': ''},
            {'foo': '',     'baz': '',      'hello': 'world4'},
        ]
        yield self.assertExported(items, header, rows,
                                  settings=settings, ordered=True)

    @defer.inlineCallbacks
    def test_export_dicts(self):
        # When dicts are used, only keys from the first row are used as
        # a header for CSV, and all fields are used for JSON Lines.
        items = [
            {'foo': 'bar', 'egg': 'spam'},
            {'foo': 'bar', 'egg': 'spam', 'baz': 'quux'},
        ]
        rows_csv = [
            {'egg': 'spam', 'foo': 'bar'},
            {'egg': 'spam', 'foo': 'bar'}
        ]
        rows_jl = items
        yield self.assertExportedCsv(items, ['egg', 'foo'], rows_csv, ordered=False)
        yield self.assertExportedJsonLines(items, rows_jl)

    @defer.inlineCallbacks
    def test_export_feed_export_fields(self):
        # FEED_EXPORT_FIELDS option allows to order export fields
        # and to select a subset of fields to export, both for Items and dicts.

        for item_cls in [self.MyItem, dict]:
            items = [
                item_cls({'foo': 'bar1', 'egg': 'spam1'}),
                item_cls({'foo': 'bar2', 'egg': 'spam2', 'baz': 'quux2'}),
            ]

            # export all columns
            settings = {'FEED_EXPORT_FIELDS': 'foo,baz,egg'}
            rows = [
                {'egg': 'spam1', 'foo': 'bar1', 'baz': ''},
                {'egg': 'spam2', 'foo': 'bar2', 'baz': 'quux2'}
            ]
            yield self.assertExported(items, ['foo', 'baz', 'egg'], rows,
                                      settings=settings, ordered=True)

            # export a subset of columns
            settings = {'FEED_EXPORT_FIELDS': 'egg,baz'}
            rows = [
                {'egg': 'spam1', 'baz': ''},
                {'egg': 'spam2', 'baz': 'quux2'}
            ]
            yield self.assertExported(items, ['egg', 'baz'], rows,
                                      settings=settings, ordered=True)

    @defer.inlineCallbacks
    def test_export_encoding(self):
        items = [dict({'foo': u'Test\xd6'})]
        header = ['foo']

        formats = {
            'json': u'[{"foo": "Test\\u00d6"}]'.encode('utf-8'),
            'jsonlines': u'{"foo": "Test\\u00d6"}\n'.encode('utf-8'),
            'xml': u'<?xml version="1.0" encoding="utf-8"?>\n<items><item><foo>Test\xd6</foo></item></items>'.encode('utf-8'),
            'csv': u'foo\r\nTest\xd6\r\n'.encode('utf-8'),
        }

        for format, expected in formats.items():
            settings = {'FEED_FORMAT': format, 'FEED_EXPORT_INDENT': None}
            data = yield self.exported_data(items, settings)
            self.assertEqual(expected, data)

        formats = {
            'json': u'[{"foo": "Test\xd6"}]'.encode('latin-1'),
            'jsonlines': u'{"foo": "Test\xd6"}\n'.encode('latin-1'),
            'xml': u'<?xml version="1.0" encoding="latin-1"?>\n<items><item><foo>Test\xd6</foo></item></items>'.encode('latin-1'),
            'csv': u'foo\r\nTest\xd6\r\n'.encode('latin-1'),
        }

        settings = {'FEED_EXPORT_INDENT': None, 'FEED_EXPORT_ENCODING': 'latin-1'}
        for format, expected in formats.items():
            settings['FEED_FORMAT'] = format
            data = yield self.exported_data(items, settings)
            self.assertEqual(expected, data)

    @defer.inlineCallbacks
    def test_export_indentation(self):
        items = [
            {'foo': ['bar']},
            {'key': 'value'},
        ]

        test_cases = [
            # JSON
            {
                'format': 'json',
                'indent': None,
                'expected': b'[{"foo": ["bar"]},{"key": "value"}]',
            },
            {
                'format': 'json',
                'indent': -1,
                'expected': b"""[
{"foo": ["bar"]},
{"key": "value"}
]""",
            },
            {
                'format': 'json',
                'indent': 0,
                'expected': b"""[
{"foo": ["bar"]},
{"key": "value"}
]""",
            },
            {
                'format': 'json',
                'indent': 2,
                'expected': b"""[
{
  "foo": [
    "bar"
  ]
},
{
  "key": "value"
}
]""",
            },
            {
                'format': 'json',
                'indent': 4,
                'expected': b"""[
{
    "foo": [
        "bar"
    ]
},
{
    "key": "value"
}
]""",
            },
            {
                'format': 'json',
                'indent': 5,
                'expected': b"""[
{
     "foo": [
          "bar"
     ]
},
{
     "key": "value"
}
]""",
            },

            # XML
            {
                'format': 'xml',
                'indent': None,
                'expected': b"""<?xml version="1.0" encoding="utf-8"?>
<items><item><foo><value>bar</value></foo></item><item><key>value</key></item></items>""",
            },
            {
                'format': 'xml',
                'indent': -1,
                'expected': b"""<?xml version="1.0" encoding="utf-8"?>
<items>
<item><foo><value>bar</value></foo></item>
<item><key>value</key></item>
</items>""",
            },
            {
                'format': 'xml',
                'indent': 0,
                'expected': b"""<?xml version="1.0" encoding="utf-8"?>
<items>
<item><foo><value>bar</value></foo></item>
<item><key>value</key></item>
</items>""",
            },
            {
                'format': 'xml',
                'indent': 2,
                'expected': b"""<?xml version="1.0" encoding="utf-8"?>
<items>
  <item>
    <foo>
      <value>bar</value>
    </foo>
  </item>
  <item>
    <key>value</key>
  </item>
</items>""",
            },
            {
                'format': 'xml',
                'indent': 4,
                'expected': b"""<?xml version="1.0" encoding="utf-8"?>
<items>
    <item>
        <foo>
            <value>bar</value>
        </foo>
    </item>
    <item>
        <key>value</key>
    </item>
</items>""",
            },
            {
                'format': 'xml',
                'indent': 5,
                'expected': b"""<?xml version="1.0" encoding="utf-8"?>
<items>
     <item>
          <foo>
               <value>bar</value>
          </foo>
     </item>
     <item>
          <key>value</key>
     </item>
</items>""",
            },
        ]

        for row in test_cases:
            settings = {'FEED_FORMAT': row['format'], 'FEED_EXPORT_INDENT': row['indent']}
            data = yield self.exported_data(items, settings)
            print(row['format'], row['indent'])
            self.assertEqual(row['expected'], data)

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
class JsonEncoderTestCase(unittest.TestCase):

    def setUp(self):
        self.encoder = ScrapyJSONEncoder()

    def test_encode_decode(self):
        dt = datetime.datetime(2010, 1, 2, 10, 11, 12)
        dts = "2010-01-02 10:11:12"
        d = datetime.date(2010, 1, 2)
        ds = "2010-01-02"
        t = datetime.time(10, 11, 12)
        ts = "10:11:12"
        dec = Decimal("1000.12")
        decs = "1000.12"
        s = {'foo'}
        ss = ['foo']
        dt_set = {dt}
        dt_sets = [dts]

        for input, output in [('foo', 'foo'), (d, ds), (t, ts), (dt, dts),
                              (dec, decs), (['foo', d], ['foo', ds]), (s, ss),
                              (dt_set, dt_sets)]:
            self.assertEqual(self.encoder.encode(input), json.dumps(output))

    def test_encode_deferred(self):
        self.assertIn('Deferred', self.encoder.encode(defer.Deferred()))

    def test_encode_request(self):
        r = Request("http://www.example.com/lala")
        rs = self.encoder.encode(r)
        self.assertIn(r.method, rs)
        self.assertIn(r.url, rs)

    def test_encode_response(self):
        r = Response("http://www.example.com/lala")
        rs = self.encoder.encode(r)
        self.assertIn(r.url, rs)
        self.assertIn(str(r.status), rs)

class TextResponseTest(BaseResponseTest):

    response_class = TextResponse

    def test_replace(self):
        super(TextResponseTest, self).test_replace()
        r1 = self.response_class("http://www.example.com", body="hello", encoding="cp852")
        r2 = r1.replace(url="http://www.example.com/other")
        r3 = r1.replace(url="http://www.example.com/other", encoding="latin1")

        assert isinstance(r2, self.response_class)
        self.assertEqual(r2.url, "http://www.example.com/other")
        self._assert_response_encoding(r2, "cp852")
        self.assertEqual(r3.url, "http://www.example.com/other")
        self.assertEqual(r3._declared_encoding(), "latin1")

    def test_unicode_url(self):
        # instantiate with unicode url without encoding (should set default encoding)
        resp = self.response_class(u"http://www.example.com/")
        self._assert_response_encoding(resp, self.response_class._DEFAULT_ENCODING)

        # make sure urls are converted to str
        resp = self.response_class(url=u"http://www.example.com/", encoding='utf-8')
        assert isinstance(resp.url, str)

        resp = self.response_class(url=u"http://www.example.com/price/\xa3", encoding='utf-8')
        self.assertEqual(resp.url, to_native_str(b'http://www.example.com/price/\xc2\xa3'))
        resp = self.response_class(url=u"http://www.example.com/price/\xa3", encoding='latin-1')
        self.assertEqual(resp.url, 'http://www.example.com/price/\xa3')
        resp = self.response_class(u"http://www.example.com/price/\xa3", headers={"Content-type": ["text/html; charset=utf-8"]})
        self.assertEqual(resp.url, to_native_str(b'http://www.example.com/price/\xc2\xa3'))
        resp = self.response_class(u"http://www.example.com/price/\xa3", headers={"Content-type": ["text/html; charset=iso-8859-1"]})
        self.assertEqual(resp.url, 'http://www.example.com/price/\xa3')

    def test_unicode_body(self):
        unicode_string = u'\u043a\u0438\u0440\u0438\u043b\u043b\u0438\u0447\u0435\u0441\u043a\u0438\u0439 \u0442\u0435\u043a\u0441\u0442'
        self.assertRaises(TypeError, self.response_class, 'http://www.example.com', body=u'unicode body')

        original_string = unicode_string.encode('cp1251')
        r1 = self.response_class('http://www.example.com', body=original_string, encoding='cp1251')

        # check body_as_unicode
        self.assertTrue(isinstance(r1.body_as_unicode(), six.text_type))
        self.assertEqual(r1.body_as_unicode(), unicode_string)

        # check response.text
        self.assertTrue(isinstance(r1.text, six.text_type))
        self.assertEqual(r1.text, unicode_string)

    def test_encoding(self):
        r1 = self.response_class("http://www.example.com", headers={"Content-type": ["text/html; charset=utf-8"]}, body=b"\xc2\xa3")
        r2 = self.response_class("http://www.example.com", encoding='utf-8', body=u"\xa3")
        r3 = self.response_class("http://www.example.com", headers={"Content-type": ["text/html; charset=iso-8859-1"]}, body=b"\xa3")
        r4 = self.response_class("http://www.example.com", body=b"\xa2\xa3")
        r5 = self.response_class("http://www.example.com", headers={"Content-type": ["text/html; charset=None"]}, body=b"\xc2\xa3")
        r6 = self.response_class("http://www.example.com", headers={"Content-type": ["text/html; charset=gb2312"]}, body=b"\xa8D")
        r7 = self.response_class("http://www.example.com", headers={"Content-type": ["text/html; charset=gbk"]}, body=b"\xa8D")

        self.assertEqual(r1._headers_encoding(), "utf-8")
        self.assertEqual(r2._headers_encoding(), None)
        self.assertEqual(r2._declared_encoding(), 'utf-8')
        self._assert_response_encoding(r2, 'utf-8')
        self.assertEqual(r3._headers_encoding(), "cp1252")
        self.assertEqual(r3._declared_encoding(), "cp1252")
        self.assertEqual(r4._headers_encoding(), None)
        self.assertEqual(r5._headers_encoding(), None)
        self._assert_response_encoding(r5, "utf-8")
        assert r4._body_inferred_encoding() is not None and r4._body_inferred_encoding() != 'ascii'
        self._assert_response_values(r1, 'utf-8', u"\xa3")
        self._assert_response_values(r2, 'utf-8', u"\xa3")
        self._assert_response_values(r3, 'iso-8859-1', u"\xa3")
        self._assert_response_values(r6, 'gb18030', u"\u2015")
        self._assert_response_values(r7, 'gb18030', u"\u2015")

        # TextResponse (and subclasses) must be passed a encoding when instantiating with unicode bodies
        self.assertRaises(TypeError, self.response_class, "http://www.example.com", body=u"\xa3")

    def test_declared_encoding_invalid(self):
        """Check that unknown declared encodings are ignored"""
        r = self.response_class("http://www.example.com",
                                headers={"Content-type": ["text/html; charset=UKNOWN"]},
                                body=b"\xc2\xa3")
        self.assertEqual(r._declared_encoding(), None)
        self._assert_response_values(r, 'utf-8', u"\xa3")

    def test_utf16(self):
        """Test utf-16 because UnicodeDammit is known to have problems with"""
        r = self.response_class("http://www.example.com",
                                body=b'\xff\xfeh\x00i\x00',
                                encoding='utf-16')
        self._assert_response_values(r, 'utf-16', u"hi")

    def test_invalid_utf8_encoded_body_with_valid_utf8_BOM(self):
        r6 = self.response_class("http://www.example.com",
                                 headers={"Content-type": ["text/html; charset=utf-8"]},
                                 body=b"\xef\xbb\xbfWORD\xe3\xab")
        self.assertEqual(r6.encoding, 'utf-8')
        self.assertEqual(r6.text, u'WORD\ufffd\ufffd')

    def test_bom_is_removed_from_body(self):
        # Inferring encoding from body also cache decoded body as sideeffect,
        # this test tries to ensure that calling response.encoding and
        # response.text in indistint order doesn't affect final
        # values for encoding and decoded body.
        url = 'http://example.com'
        body = b"\xef\xbb\xbfWORD"
        headers = {"Content-type": ["text/html; charset=utf-8"]}

        # Test response without content-type and BOM encoding
        response = self.response_class(url, body=body)
        self.assertEqual(response.encoding, 'utf-8')
        self.assertEqual(response.text, u'WORD')
        response = self.response_class(url, body=body)
        self.assertEqual(response.text, u'WORD')
        self.assertEqual(response.encoding, 'utf-8')

        # Body caching sideeffect isn't triggered when encoding is declared in
        # content-type header but BOM still need to be removed from decoded
        # body
        response = self.response_class(url, headers=headers, body=body)
        self.assertEqual(response.encoding, 'utf-8')
        self.assertEqual(response.text, u'WORD')
        response = self.response_class(url, headers=headers, body=body)
        self.assertEqual(response.text, u'WORD')
        self.assertEqual(response.encoding, 'utf-8')

    def test_replace_wrong_encoding(self):
        """Test invalid chars are replaced properly"""
        r = self.response_class("http://www.example.com", encoding='utf-8', body=b'PREFIX\xe3\xabSUFFIX')
        # XXX: Policy for replacing invalid chars may suffer minor variations
        # but it should always contain the unicode replacement char (u'\ufffd')
        assert u'\ufffd' in r.text, repr(r.text)
        assert u'PREFIX' in r.text, repr(r.text)
        assert u'SUFFIX' in r.text, repr(r.text)

        # Do not destroy html tags due to encoding bugs
        r = self.response_class("http://example.com", encoding='utf-8', \
                body=b'\xf0<span>value</span>')
        assert u'<span>value</span>' in r.text, repr(r.text)

        # FIXME: This test should pass once we stop using BeautifulSoup's UnicodeDammit in TextResponse
        #r = self.response_class("http://www.example.com", body=b'PREFIX\xe3\xabSUFFIX')
        #assert u'\ufffd' in r.text, repr(r.text)

    def test_selector(self):
        body = b"<html><head><title>Some page</title><body></body></html>"
        response = self.response_class("http://www.example.com", body=body)

        self.assertIsInstance(response.selector, Selector)
        self.assertEqual(response.selector.type, 'html')
        self.assertIs(response.selector, response.selector)  # property is cached
        self.assertIs(response.selector.response, response)

        self.assertEqual(
            response.selector.xpath("//title/text()").extract(),
            [u'Some page']
        )
        self.assertEqual(
            response.selector.css("title::text").extract(),
            [u'Some page']
        )
        self.assertEqual(
            response.selector.re("Some (.*)</title>"),
            [u'page']
        )

    def test_selector_shortcuts(self):
        body = b"<html><head><title>Some page</title><body></body></html>"
        response = self.response_class("http://www.example.com", body=body)

        self.assertEqual(
            response.xpath("//title/text()").extract(),
            response.selector.xpath("//title/text()").extract(),
        )
        self.assertEqual(
            response.css("title::text").extract(),
            response.selector.css("title::text").extract(),
        )

    def test_selector_shortcuts_kwargs(self):
        body = b"<html><head><title>Some page</title><body><p class=\"content\">A nice paragraph.</p></body></html>"
        response = self.response_class("http://www.example.com", body=body)

        self.assertEqual(
            response.xpath("normalize-space(//p[@class=$pclass])", pclass="content").extract(),
            response.xpath("normalize-space(//p[@class=\"content\"])").extract(),
        )
        self.assertEqual(
            response.xpath("//title[count(following::p[@class=$pclass])=$pcount]/text()",
                pclass="content", pcount=1).extract(),
            response.xpath("//title[count(following::p[@class=\"content\"])=1]/text()").extract(),
        )

    def test_urljoin_with_base_url(self):
        """Test urljoin shortcut which also evaluates base-url through get_base_url()."""
        body = b'<html><body><base href="https://example.net"></body></html>'
        joined = self.response_class('http://www.example.com', body=body).urljoin('/test')
        absolute = 'https://example.net/test'
        self.assertEqual(joined, absolute)

        body = b'<html><body><base href="/elsewhere"></body></html>'
        joined = self.response_class('http://www.example.com', body=body).urljoin('test')
        absolute = 'http://www.example.com/test'
        self.assertEqual(joined, absolute)

        body = b'<html><body><base href="/elsewhere/"></body></html>'
        joined = self.response_class('http://www.example.com', body=body).urljoin('test')
        absolute = 'http://www.example.com/elsewhere/test'
        self.assertEqual(joined, absolute)

    def test_follow_selector(self):
        resp = self._links_response()
        urls = [
            'http://example.com/sample2.html',
            'http://example.com/sample3.html',
            'http://example.com/sample3.html',
            'http://example.com/sample3.html#foo',
            'http://www.google.com/something',
            'http://example.com/innertag.html'
        ]

        # select <a> elements
        for sellist in [resp.css('a'), resp.xpath('//a')]:
            for sel, url in zip(sellist, urls):
                self._assert_followed_url(sel, url, response=resp)

        # select <link> elements
        self._assert_followed_url(
            Selector(text='<link href="foo"></link>').css('link')[0],
            'http://example.com/foo',
            response=resp
        )

        # href attributes should work
        for sellist in [resp.css('a::attr(href)'), resp.xpath('//a/@href')]:
            for sel, url in zip(sellist, urls):
                self._assert_followed_url(sel, url, response=resp)

        # non-a elements are not supported
        self.assertRaises(ValueError, resp.follow, resp.css('div')[0])

    def test_follow_selector_list(self):
        resp = self._links_response()
        self.assertRaisesRegexp(ValueError, 'SelectorList',
                                resp.follow, resp.css('a'))

    def test_follow_selector_invalid(self):
        resp = self._links_response()
        self.assertRaisesRegexp(ValueError, 'Unsupported',
                                resp.follow, resp.xpath('count(//div)')[0])

    def test_follow_selector_attribute(self):
        resp = self._links_response()
        for src in resp.css('img::attr(src)'):
            self._assert_followed_url(src, 'http://example.com/sample2.jpg')

    def test_follow_selector_no_href(self):
        resp = self.response_class(
            url='http://example.com',
            body=b'<html><body><a name=123>click me</a></body></html>',
        )
        self.assertRaisesRegexp(ValueError, 'no href',
                                resp.follow, resp.css('a')[0])

    def test_follow_whitespace_selector(self):
        resp = self.response_class(
            'http://example.com',
            body=b'''<html><body><a href=" foo\n">click me</a></body></html>'''
        )
        self._assert_followed_url(resp.css('a')[0],
                                 'http://example.com/foo',
                                  response=resp)
        self._assert_followed_url(resp.css('a::attr(href)')[0],
                                 'http://example.com/foo',
                                  response=resp)

    def test_follow_encoding(self):
        resp1 = self.response_class(
            'http://example.com',
            encoding='utf8',
            body=u'<html><body><a href="foo?привет">click me</a></body></html>'.encode('utf8')
        )
        req = self._assert_followed_url(
            resp1.css('a')[0],
            'http://example.com/foo?%D0%BF%D1%80%D0%B8%D0%B2%D0%B5%D1%82',
            response=resp1,
        )
        self.assertEqual(req.encoding, 'utf8')

        resp2 = self.response_class(
            'http://example.com',
            encoding='cp1251',
            body=u'<html><body><a href="foo?привет">click me</a></body></html>'.encode('cp1251')
        )
        req = self._assert_followed_url(
            resp2.css('a')[0],
            'http://example.com/foo?%EF%F0%E8%E2%E5%F2',
            response=resp2,
        )
        self.assertEqual(req.encoding, 'cp1251')


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


class ImageDownloadCrawlTestCase(FileDownloadCrawlTestCase):
    pipeline_class = 'scrapy.pipelines.images.ImagesPipeline'
    store_setting_key = 'IMAGES_STORE'
    media_key = 'images'
    media_urls_key = 'image_urls'

    # somehow checksums for images are different for Python 3.3
    expected_checksums = None

class MarshalFifoDiskQueueTest(t.FifoDiskQueueTest):

    chunksize = 100000

    def queue(self):
        return MarshalFifoDiskQueue(self.qpath, chunksize=self.chunksize)

    def test_serialize(self):
        q = self.queue()
        q.push('a')
        q.push(123)
        q.push({'a': 'dict'})
        self.assertEqual(q.pop(), 'a')
        self.assertEqual(q.pop(), 123)
        self.assertEqual(q.pop(), {'a': 'dict'})

    test_nonserializable_object = nonserializable_object_test
