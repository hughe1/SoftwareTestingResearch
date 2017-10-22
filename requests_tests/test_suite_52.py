# -*- coding: utf-8 -*-

"""Tests for Requests."""

from __future__ import division
import json
import os
import pickle
import collections
import contextlib
import warnings

import io
import requests
import pytest
from requests.adapters import HTTPAdapter
from requests.auth import HTTPDigestAuth, _basic_auth_str
from requests.compat import (
    Morsel, cookielib, getproxies, str, urlparse,
    builtin_str, OrderedDict)
from requests.cookies import (
    cookiejar_from_dict, morsel_to_cookie)
from requests.exceptions import (
    ConnectionError, ConnectTimeout, InvalidSchema, InvalidURL,
    MissingSchema, ReadTimeout, Timeout, RetryError, TooManyRedirects,
    ProxyError, InvalidHeader, UnrewindableBodyError, SSLError)
from requests.models import PreparedRequest
from requests.structures import CaseInsensitiveDict
from requests.sessions import SessionRedirectMixin
from requests.models import urlencode
from requests.hooks import default_hooks

# from .compat import StringIO, u
# from .utils import override_environ
from urllib3.util import Timeout as Urllib3Timeout

# Requests to this URL should always fail with a connection timeout (nothing
# listening on that port)
TARPIT = 'http://10.255.255.1'

try:
    from ssl import SSLContext
    del SSLContext
    HAS_MODERN_SSL = True
except ImportError:
    HAS_MODERN_SSL = False

try:
    requests.pyopenssl
    HAS_PYOPENSSL = True
except AttributeError:
    HAS_PYOPENSSL = False

# -*- coding: utf-8 -*-

import threading
import socket
import time

import pytest
import requests
from testserver.server import Server

# -*- coding: utf-8 -*-

import pytest

from requests.structures import CaseInsensitiveDict, LookupDict

# -*- coding: utf-8 -*-

import os
import copy
from io import BytesIO

import pytest
from requests import compat
from requests.cookies import RequestsCookieJar
from requests.structures import CaseInsensitiveDict
from requests.utils import (
    address_in_network, dotted_netmask,
    get_auth_from_url, get_encoding_from_headers,
    get_encodings_from_content, get_environ_proxies,
    guess_filename, guess_json_utf, is_ipv4_address,
    is_valid_cidr, iter_slices, parse_dict_header,
    parse_header_links, prepend_scheme_if_needed,
    requote_uri, select_proxy, should_bypass_proxies, super_len,
    to_key_val_list, to_native_string,
    unquote_header_value, unquote_unreserved,
    urldefragauth, add_dict_to_cookiejar, set_environ)
from requests._internal_utils import unicode_is_ascii


# -*- encoding: utf-8

import sys

import pytest

from requests.help import info

# -*- coding: utf-8 -*-

import pytest

from requests import hooks


class TestTimeout:

    def test_stream_timeout(self, httpbin):
        try:
            requests.get(httpbin('delay/10'), timeout=2.0)
        except requests.exceptions.Timeout as e:
            assert 'Read timed out' in e.args[0].args[0]

    @pytest.mark.parametrize(
        'timeout, error_text', (
            ((3, 4, 5), '(connect, read)'),
            ('foo', 'must be an int, float or None'),
        ))
    def test_invalid_timeout(self, httpbin, timeout, error_text):
        with pytest.raises(ValueError) as e:
            requests.get(httpbin('get'), timeout=timeout)
        assert error_text in str(e)

    @pytest.mark.parametrize(
        'timeout', (
            None,
            Urllib3Timeout(connect=None, read=None)
        ))
    def test_none_timeout(self, httpbin, timeout):
        """Check that you can set None as a valid timeout value.

        To actually test this behavior, we'd want to check that setting the
        timeout to None actually lets the request block past the system default
        timeout. However, this would make the test suite unbearably slow.
        Instead we verify that setting the timeout to None does not prevent the
        request from succeeding.
        """
        r = requests.get(httpbin('get'), timeout=timeout)
        assert r.status_code == 200

    @pytest.mark.parametrize(
        'timeout', (
            (None, 0.1),
            Urllib3Timeout(connect=None, read=0.1)
        ))
    def test_read_timeout(self, httpbin, timeout):
        try:
            requests.get(httpbin('delay/10'), timeout=timeout)
            pytest.fail('The recv() request should time out.')
        except ReadTimeout:
            pass

    @pytest.mark.parametrize(
        'timeout', (
            (0.1, None),
            Urllib3Timeout(connect=0.1, read=None)
        ))
    def test_connect_timeout(self, timeout):
        try:
            requests.get(TARPIT, timeout=timeout)
            pytest.fail('The connect() request should time out.')
        except ConnectTimeout as e:
            assert isinstance(e, ConnectionError)
            assert isinstance(e, Timeout)

    @pytest.mark.parametrize(
        'timeout', (
            (0.1, 0.1),
            Urllib3Timeout(connect=0.1, read=0.1)
        ))
    def test_total_timeout_connect(self, timeout):
        try:
            requests.get(TARPIT, timeout=timeout)
            pytest.fail('The connect() request should time out.')
        except ConnectTimeout:
            pass

    def test_encoded_methods(self, httpbin):
        """See: https://github.com/requests/requests/issues/2316"""
        r = requests.request(b'GET', httpbin('get'))
        assert r.ok


SendCall = collections.namedtuple('SendCall', ('args', 'kwargs'))



def test_json_encodes_as_bytes():
    # urllib3 expects bodies as bytes-like objects
    body = {"key": "value"}
    p = PreparedRequest()
    p.prepare(
        method='GET',
        url='https://www.example.com/',
        json=body
    )
    assert isinstance(p.body, bytes)


def test_requests_are_updated_each_time(httpbin):
    session = RedirectSession([303, 307])
    prep = requests.Request('POST', httpbin('post')).prepare()
    r0 = session.send(prep)
    assert r0.request.method == 'POST'
    assert session.calls[-1] == SendCall((r0.request,), {})
    redirect_generator = session.resolve_redirects(r0, prep)
    default_keyword_args = {
        'stream': False,
        'verify': True,
        'cert': None,
        'timeout': None,
        'allow_redirects': False,
        'proxies': {},
    }
    for response in redirect_generator:
        assert response.request.method == 'GET'
        send_call = SendCall((response.request,), default_keyword_args)
        assert session.calls[-1] == send_call


# @pytest.mark.parametrize("var,url,proxy", [
#     ('http_proxy', 'http://example.com', 'socks5://proxy.com:9876'),
#     ('https_proxy', 'https://example.com', 'socks5://proxy.com:9876'),
#     ('all_proxy', 'http://example.com', 'socks5://proxy.com:9876'),
#     ('all_proxy', 'https://example.com', 'socks5://proxy.com:9876'),
# ])
# def test_proxy_env_vars_override_default(var, url, proxy):
#     session = requests.Session()
#     prep = PreparedRequest()
#     prep.prepare(method='GET', url=url)
#
#     kwargs = {
#         var: proxy
#     }
#     scheme = urlparse(url).scheme
#     with override_environ(**kwargs):
#         proxies = session.rebuild_proxies(prep, {})
#         assert scheme in proxies
#         assert proxies[scheme] == proxy


@pytest.mark.parametrize(
    'data', (
        (('a', 'b'), ('c', 'd')),
        (('c', 'd'), ('a', 'b')),
        (('a', 'b'), ('c', 'd'), ('e', 'f')),
    ))
def test_data_argument_accepts_tuples(data):
    """Ensure that the data argument will accept tuples of strings
    and properly encode them.
    """
    p = PreparedRequest()
    p.prepare(
        method='GET',
        url='http://www.example.com',
        data=data,
        hooks=default_hooks()
    )
    assert p.body == urlencode(data)


# @pytest.mark.parametrize(
#     'kwargs', (
#         None,
#         {
#             'method': 'GET',
#             'url': 'http://www.example.com',
#             'data': 'foo=bar',
#             'hooks': default_hooks()
#         },
#         {
#             'method': 'GET',
#             'url': 'http://www.example.com',
#             'data': 'foo=bar',
#             'hooks': default_hooks(),
#             'cookies': {'foo': 'bar'}
#         },
#         {
#             'method': 'GET',
#             'url': u('http://www.example.com/üniçø∂é')
#         },
#     ))
# def test_prepared_copy(kwargs):
#     p = PreparedRequest()
#     if kwargs:
#         p.prepare(**kwargs)
#     copy = p.copy()
#     for attr in ('method', 'url', 'headers', '_cookies', 'body', 'hooks'):
#         assert getattr(p, attr) == getattr(copy, attr)


def test_urllib3_retries(httpbin):
    from urllib3.util import Retry
    s = requests.Session()
    s.mount('http://', HTTPAdapter(max_retries=Retry(
        total=2, status_forcelist=[500]
    )))

    with pytest.raises(RetryError):
        s.get(httpbin('status/500'))


# def test_urllib3_pool_connection_closed(httpbin):
#     s = requests.Session()
#     s.mount('http://', HTTPAdapter(pool_connections=0, pool_maxsize=0))
#
#     try:
#         s.get(httpbin('status/200'))
#     except ConnectionError as e:
#         assert u"Pool is closed." in str(e)


# class TestPreparingURLs(object):
#     @pytest.mark.parametrize(
#         'url,expected',
#         (
#             ('http://google.com', 'http://google.com/'),
#             (u'http://ジェーピーニック.jp', u'http://xn--hckqz9bzb1cyrb.jp/'),
#             (u'http://xn--n3h.net/', u'http://xn--n3h.net/'),
#             (
#                 u'http://ジェーピーニック.jp'.encode('utf-8'),
#                 u'http://xn--hckqz9bzb1cyrb.jp/'
#             ),
#             (
#                 u'http://straße.de/straße',
#                 u'http://xn--strae-oqa.de/stra%C3%9Fe'
#             ),
#             (
#                 u'http://straße.de/straße'.encode('utf-8'),
#                 u'http://xn--strae-oqa.de/stra%C3%9Fe'
#             ),
#             (
#                 u'http://Königsgäßchen.de/straße',
#                 u'http://xn--knigsgchen-b4a3dun.de/stra%C3%9Fe'
#             ),
#             (
#                 u'http://Königsgäßchen.de/straße'.encode('utf-8'),
#                 u'http://xn--knigsgchen-b4a3dun.de/stra%C3%9Fe'
#             ),
#             (
#                 b'http://xn--n3h.net/',
#                 u'http://xn--n3h.net/'
#             ),
#             (
#                 b'http://[1200:0000:ab00:1234:0000:2552:7777:1313]:12345/',
#                 u'http://[1200:0000:ab00:1234:0000:2552:7777:1313]:12345/'
#             ),
#             (
#                 u'http://[1200:0000:ab00:1234:0000:2552:7777:1313]:12345/',
#                 u'http://[1200:0000:ab00:1234:0000:2552:7777:1313]:12345/'
#             )
#         )
#     )
#     def test_preparing_url(self, url, expected):
#         r = requests.Request('GET', url=url)
#         p = r.prepare()
#         assert p.url == expected

    # @pytest.mark.parametrize(
    #     'url',
    #     (
    #         b"http://*.google.com",
    #         b"http://*",
    #         u"http://*.google.com",
    #         u"http://*",
    #         u"http://☃.net/"
    #     )
    # )
    # def test_preparing_bad_url(self, url):
    #     r = requests.Request('GET', url=url)
    #     with pytest.raises(requests.exceptions.InvalidURL):
    #         r.prepare()

    # @pytest.mark.parametrize(
    #     'input, expected',
    #     (
    #         (
    #             b"http+unix://%2Fvar%2Frun%2Fsocket/path%7E",
    #             u"http+unix://%2Fvar%2Frun%2Fsocket/path~",
    #         ),
    #         (
    #             u"http+unix://%2Fvar%2Frun%2Fsocket/path%7E",
    #             u"http+unix://%2Fvar%2Frun%2Fsocket/path~",
    #         ),
    #         (
    #             b"mailto:user@example.org",
    #             u"mailto:user@example.org",
    #         ),
    #         (
    #             u"mailto:user@example.org",
    #             u"mailto:user@example.org",
    #         ),
    #         (
    #             b"data:SSDimaUgUHl0aG9uIQ==",
    #             u"data:SSDimaUgUHl0aG9uIQ==",
    #         )
    #     )
    # )
    # def test_url_mutation(self, input, expected):
    #     """
    #     This test validates that we correctly exclude some URLs from
    #     preparation, and that we handle others. Specifically, it tests that
    #     any URL whose scheme doesn't begin with "http" is left alone, and
    #     those whose scheme *does* begin with "http" are mutated.
    #     """
    #     r = requests.Request('GET', url=input)
    #     p = r.prepare()
    #     assert p.url == expected

    # @pytest.mark.parametrize(
    #     'input, params, expected',
    #     (
    #         (
    #             b"http+unix://%2Fvar%2Frun%2Fsocket/path",
    #             {"key": "value"},
    #             u"http+unix://%2Fvar%2Frun%2Fsocket/path?key=value",
    #         ),
    #         (
    #             u"http+unix://%2Fvar%2Frun%2Fsocket/path",
    #             {"key": "value"},
    #             u"http+unix://%2Fvar%2Frun%2Fsocket/path?key=value",
    #         ),
    #         (
    #             b"mailto:user@example.org",
    #             {"key": "value"},
    #             u"mailto:user@example.org",
    #         ),
    #         (
    #             u"mailto:user@example.org",
    #             {"key": "value"},
    #             u"mailto:user@example.org",
    #         ),
    #     )
    # )
    # def test_parameters_for_nonstandard_schemes(self, input, params, expected):
    #     """
    #     Setting parameters for nonstandard schemes is allowed if those schemes
    #     begin with "http", and is forbidden otherwise.
    #     """
    #     r = requests.Request('GET', url=input, params=params)
    #     p = r.prepare()
    #     assert p.url == expected

class TestIsValidCIDR:

    def test_valid(self):
        assert is_valid_cidr('192.168.1.0/24')

    @pytest.mark.parametrize(
        'value', (
            '8.8.8.8',
            '192.168.1.0/a',
            '192.168.1.0/128',
            '192.168.1.0/-1',
            '192.168.1.999/24',
        ))
    def test_invalid(self, value):
        assert not is_valid_cidr(value)


class TestToKeyValList:

    @pytest.mark.parametrize(
        'value, expected', (
            ([('key', 'val')], [('key', 'val')]),
            ((('key', 'val'), ), [('key', 'val')]),
            ({'key': 'val'}, [('key', 'val')]),
            (None, None)
        ))
    def test_valid(self, value, expected):
        assert to_key_val_list(value) == expected

    def test_invalid(self):
        with pytest.raises(ValueError):
            to_key_val_list('string')

# -*- coding: utf-8 -*-

import pytest
import threading
import requests

from testserver.server import Server, consume_socket_content



def test_can_access_urllib3_attribute():
    requests.packages.urllib3


@pytest.mark.skipif(sys.version_info < (2,7), reason="Only run on Python 2.7+")
def test_system_ssl():
    """Verify we're actually setting system_ssl when it should be available."""
    assert info()['system_ssl']['version'] != ''


class TestToKeyValList:

    @pytest.mark.parametrize(
        'value, expected', (
            ([('key', 'val')], [('key', 'val')]),
            ((('key', 'val'), ), [('key', 'val')]),
            ({'key': 'val'}, [('key', 'val')]),
            (None, None)
        ))
    def test_valid(self, value, expected):
        assert to_key_val_list(value) == expected

    def test_invalid(self):
        with pytest.raises(ValueError):
            to_key_val_list('string')


class TestUnquoteHeaderValue:

    @pytest.mark.parametrize(
        'value, expected', (
            (None, None),
            ('Test', 'Test'),
            ('"Test"', 'Test'),
            ('"Test\\\\"', 'Test\\'),
            ('"\\\\Comp\\Res"', '\\Comp\\Res'),
        ))
    def test_valid(self, value, expected):
        assert unquote_header_value(value) == expected

    def test_is_filename(self):
        assert unquote_header_value('"\\\\Comp\\Res"', True) == '\\\\Comp\\Res'


class TestToKeyValList:

    @pytest.mark.parametrize(
        'value, expected', (
            ([('key', 'val')], [('key', 'val')]),
            ((('key', 'val'), ), [('key', 'val')]),
            ({'key': 'val'}, [('key', 'val')]),
            (None, None)
        ))
    def test_valid(self, value, expected):
        assert to_key_val_list(value) == expected

    def test_invalid(self):
        with pytest.raises(ValueError):
            to_key_val_list('string')

