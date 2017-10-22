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


@pytest.mark.skipif(sys.version_info < (2,7), reason="Only run on Python 2.7+")
def test_system_ssl():
    """Verify we're actually setting system_ssl when it should be available."""
    assert info()['system_ssl']['version'] != ''


def test_digestauth_only_on_4xx():
    """Ensure we only send digestauth on 4xx challenges.

    See https://github.com/requests/requests/issues/3772.
    """
    text_200_chal = (b'HTTP/1.1 200 OK\r\n'
                     b'Content-Length: 0\r\n'
                     b'WWW-Authenticate: Digest nonce="6bf5d6e4da1ce66918800195d6b9130d"'
                     b', opaque="372825293d1c26955496c80ed6426e9e", '
                     b'realm="me@kennethreitz.com", qop=auth\r\n\r\n')

    auth = requests.auth.HTTPDigestAuth('user', 'pass')

    def digest_response_handler(sock):
        # Respond to GET with a 200 containing www-authenticate header.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert request_content.startswith(b"GET / HTTP/1.1")
        sock.send(text_200_chal)

        # Verify the client didn't respond with auth.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert request_content == b''

        return request_content

    close_server = threading.Event()
    server = Server(digest_response_handler, wait_to_close_event=close_server)

    with server as (host, port):
        url = 'http://{0}:{1}/'.format(host, port)
        r = requests.get(url, auth=auth)
        # Verify server didn't receive auth from us.
        assert r.status_code == 200
        assert len(r.history) == 0
        close_server.set()


_schemes_by_var_prefix = [
    ('http', ['http']),
    ('https', ['https']),
    ('all', ['http', 'https']),
]




def test_digestauth_401_count_reset_on_redirect():
    """Ensure we correctly reset num_401_calls after a successful digest auth,
    followed by a 302 redirect to another digest auth prompt.

    See https://github.com/requests/requests/issues/1979.
    """
    text_401 = (b'HTTP/1.1 401 UNAUTHORIZED\r\n'
                b'Content-Length: 0\r\n'
                b'WWW-Authenticate: Digest nonce="6bf5d6e4da1ce66918800195d6b9130d"'
                b', opaque="372825293d1c26955496c80ed6426e9e", '
                b'realm="me@kennethreitz.com", qop=auth\r\n\r\n')

    text_302 = (b'HTTP/1.1 302 FOUND\r\n'
                b'Content-Length: 0\r\n'
                b'Location: /\r\n\r\n')

    text_200 = (b'HTTP/1.1 200 OK\r\n'
                b'Content-Length: 0\r\n\r\n')

    expected_digest = (b'Authorization: Digest username="user", '
                       b'realm="me@kennethreitz.com", '
                       b'nonce="6bf5d6e4da1ce66918800195d6b9130d", uri="/"')

    auth = requests.auth.HTTPDigestAuth('user', 'pass')

    def digest_response_handler(sock):
        # Respond to initial GET with a challenge.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert request_content.startswith(b"GET / HTTP/1.1")
        sock.send(text_401)

        # Verify we receive an Authorization header in response, then redirect.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert expected_digest in request_content
        sock.send(text_302)

        # Verify Authorization isn't sent to the redirected host,
        # then send another challenge.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert b'Authorization:' not in request_content
        sock.send(text_401)

        # Verify Authorization is sent correctly again, and return 200 OK.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert expected_digest in request_content
        sock.send(text_200)

        return request_content

    close_server = threading.Event()
    server = Server(digest_response_handler, wait_to_close_event=close_server)

    with server as (host, port):
        url = 'http://{0}:{1}/'.format(host, port)
        r = requests.get(url, auth=auth)
        # Verify server succeeded in authenticating.
        assert r.status_code == 200
        # Verify Authorization was sent in final request.
        assert 'Authorization' in r.request.headers
        assert r.request.headers['Authorization'].startswith('Digest ')
        # Verify redirect happened as we expected.
        assert r.history[0].status_code == 302
        close_server.set()


def test_digestauth_only_on_4xx():
    """Ensure we only send digestauth on 4xx challenges.

    See https://github.com/requests/requests/issues/3772.
    """
    text_200_chal = (b'HTTP/1.1 200 OK\r\n'
                     b'Content-Length: 0\r\n'
                     b'WWW-Authenticate: Digest nonce="6bf5d6e4da1ce66918800195d6b9130d"'
                     b', opaque="372825293d1c26955496c80ed6426e9e", '
                     b'realm="me@kennethreitz.com", qop=auth\r\n\r\n')

    auth = requests.auth.HTTPDigestAuth('user', 'pass')

    def digest_response_handler(sock):
        # Respond to GET with a 200 containing www-authenticate header.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert request_content.startswith(b"GET / HTTP/1.1")
        sock.send(text_200_chal)

        # Verify the client didn't respond with auth.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert request_content == b''

        return request_content

    close_server = threading.Event()
    server = Server(digest_response_handler, wait_to_close_event=close_server)

    with server as (host, port):
        url = 'http://{0}:{1}/'.format(host, port)
        r = requests.get(url, auth=auth)
        # Verify server didn't receive auth from us.
        assert r.status_code == 200
        assert len(r.history) == 0
        close_server.set()


_schemes_by_var_prefix = [
    ('http', ['http']),
    ('https', ['https']),
    ('all', ['http', 'https']),
]




class TestGuessJSONUTF:

    @pytest.mark.parametrize(
        'encoding', (
            'utf-32', 'utf-8-sig', 'utf-16', 'utf-8', 'utf-16-be', 'utf-16-le',
            'utf-32-be', 'utf-32-le'
        ))
    def test_encoded(self, encoding):
        data = '{}'.encode(encoding)
        assert guess_json_utf(data) == encoding

    def test_bad_utf_like_encoding(self):
        assert guess_json_utf(b'\x00\x00\x00\x00') is None

    @pytest.mark.parametrize(
        ('encoding', 'expected'), (
            ('utf-16-be', 'utf-16'),
            ('utf-16-le', 'utf-16'),
            ('utf-32-be', 'utf-32'),
            ('utf-32-le', 'utf-32')
        ))
    def test_guess_by_bom(self, encoding, expected):
        data = u'\ufeff{}'.encode(encoding)
        assert guess_json_utf(data) == expected


USER = PASSWORD = "%!*'();:@&=+$,/?#[] "
ENCODED_USER = compat.quote(USER, '')
ENCODED_PASSWORD = compat.quote(PASSWORD, '')


@pytest.mark.parametrize(
    'url, auth', (
        (
            'http://' + ENCODED_USER + ':' + ENCODED_PASSWORD + '@' +
            'request.com/url.html#test',
            (USER, PASSWORD)
        ),
        (
            'http://user:pass@complex.url.com/path?query=yes',
            ('user', 'pass')
        ),
        (
            'http://user:pass%20pass@complex.url.com/path?query=yes',
            ('user', 'pass pass')
        ),
        (
            'http://user:pass pass@complex.url.com/path?query=yes',
            ('user', 'pass pass')
        ),
        (
            'http://user%25user:pass@complex.url.com/path?query=yes',
            ('user%user', 'pass')
        ),
        (
            'http://user:pass%23pass@complex.url.com/path?query=yes',
            ('user', 'pass#pass')
        ),
        (
            'http://complex.url.com/path?query=yes',
            ('', '')
        ),
    ))
def test_get_auth_from_url(url, auth):
    assert get_auth_from_url(url) == auth


@pytest.mark.parametrize(
    'uri, expected', (
        (
            # Ensure requoting doesn't break expectations
            'http://example.com/fiz?buz=%25ppicture',
            'http://example.com/fiz?buz=%25ppicture',
        ),
        (
            # Ensure we handle unquoted percent signs in redirects
            'http://example.com/fiz?buz=%ppicture',
            'http://example.com/fiz?buz=%25ppicture',
        ),
    ))
def test_requote_uri_with_unquoted_percents(uri, expected):
    """See: https://github.com/requests/requests/issues/2356"""
    assert requote_uri(uri) == expected


@pytest.mark.parametrize(
    'uri, expected', (
        (
            # Illegal bytes
            'http://example.com/?a=%--',
            'http://example.com/?a=%--',
        ),
        (
            # Reserved characters
            'http://example.com/?a=%300',
            'http://example.com/?a=00',
        )
    ))
def test_unquote_unreserved(uri, expected):
    assert unquote_unreserved(uri) == expected


@pytest.mark.parametrize(
    'mask, expected', (
        (8, '255.0.0.0'),
        (24, '255.255.255.0'),
        (25, '255.255.255.128'),
    ))
def test_dotted_netmask(mask, expected):
    assert dotted_netmask(mask) == expected


http_proxies = {'http': 'http://http.proxy',
                'http://some.host': 'http://some.host.proxy'}
all_proxies = {'all': 'socks5://http.proxy',
               'all://some.host': 'socks5://some.host.proxy'}
mixed_proxies = {'http': 'http://http.proxy',
                 'http://some.host': 'http://some.host.proxy',
                 'all': 'socks5://http.proxy'}
@pytest.mark.parametrize(
    'url, expected, proxies', (
        ('hTTp://u:p@Some.Host/path', 'http://some.host.proxy', http_proxies),
        ('hTTp://u:p@Other.Host/path', 'http://http.proxy', http_proxies),
        ('hTTp:///path', 'http://http.proxy', http_proxies),
        ('hTTps://Other.Host', None, http_proxies),
        ('file:///etc/motd', None, http_proxies),

        ('hTTp://u:p@Some.Host/path', 'socks5://some.host.proxy', all_proxies),
        ('hTTp://u:p@Other.Host/path', 'socks5://http.proxy', all_proxies),
        ('hTTp:///path', 'socks5://http.proxy', all_proxies),
        ('hTTps://Other.Host', 'socks5://http.proxy', all_proxies),

        ('http://u:p@other.host/path', 'http://http.proxy', mixed_proxies),
        ('http://u:p@some.host/path', 'http://some.host.proxy', mixed_proxies),
        ('https://u:p@other.host/path', 'socks5://http.proxy', mixed_proxies),
        ('https://u:p@some.host/path', 'socks5://http.proxy', mixed_proxies),
        ('https://', 'socks5://http.proxy', mixed_proxies),
        # XXX: unsure whether this is reasonable behavior
        ('file:///etc/motd', 'socks5://http.proxy', all_proxies),
    ))
def test_select_proxies(url, expected, proxies):
    """Make sure we can select per-host proxies correctly."""
    assert select_proxy(url, proxies) == expected


@pytest.mark.parametrize(
    'value, expected', (
        ('foo="is a fish", bar="as well"', {'foo': 'is a fish', 'bar': 'as well'}),
        ('key_without_value', {'key_without_value': None})
    ))
def test_parse_dict_header(value, expected):
    assert parse_dict_header(value) == expected


@pytest.mark.parametrize(
    'value, expected', (
        (
            CaseInsensitiveDict(),
            None
        ),
        (
            CaseInsensitiveDict({'content-type': 'application/json; charset=utf-8'}),
            'utf-8'
        ),
        (
            CaseInsensitiveDict({'content-type': 'text/plain'}),
            'ISO-8859-1'
        ),
    ))
def test_get_encoding_from_headers(value, expected):
    assert get_encoding_from_headers(value) == expected


@pytest.mark.parametrize(
    'value, length', (
        ('', 0),
        ('T', 1),
        ('Test', 4),
        ('Cont', 0),
        ('Other', -5),
        ('Content', None),
    ))
def test_iter_slices(value, length):
    if length is None or (length <= 0 and len(value) > 0):
        # Reads all content at once
        assert len(list(iter_slices(value, length))) == 1
    else:
        assert len(list(iter_slices(value, 1))) == length


@pytest.mark.parametrize(
    'value, expected', (
        (
            '<http:/.../front.jpeg>; rel=front; type="image/jpeg"',
            [{'url': 'http:/.../front.jpeg', 'rel': 'front', 'type': 'image/jpeg'}]
        ),
        (
            '<http:/.../front.jpeg>',
            [{'url': 'http:/.../front.jpeg'}]
        ),
        (
            '<http:/.../front.jpeg>;',
            [{'url': 'http:/.../front.jpeg'}]
        ),
        (
            '<http:/.../front.jpeg>; type="image/jpeg",<http://.../back.jpeg>;',
            [
                {'url': 'http:/.../front.jpeg', 'type': 'image/jpeg'},
                {'url': 'http://.../back.jpeg'}
            ]
        ),
        (
            '',
            []
        ),
    ))
def test_parse_header_links(value, expected):
    assert parse_header_links(value) == expected


@pytest.mark.parametrize(
    'value, expected', (
        ('example.com/path', 'http://example.com/path'),
        ('//example.com/path', 'http://example.com/path'),
    ))
def test_prepend_scheme_if_needed(value, expected):
    assert prepend_scheme_if_needed(value, 'http') == expected


@pytest.mark.parametrize(
    'value, expected', (
        ('T', 'T'),
        (b'T', 'T'),
        (u'T', 'T'),
    ))
def test_to_native_string(value, expected):
    assert to_native_string(value) == expected


@pytest.mark.parametrize(
    'url, expected', (
        ('http://u:p@example.com/path?a=1#test', 'http://example.com/path?a=1'),
        ('http://example.com/path', 'http://example.com/path'),
        ('//u:p@example.com/path', '//example.com/path'),
        ('//example.com/path', '//example.com/path'),
        ('example.com/path', '//example.com/path'),
        ('scheme:u:p@example.com/path', 'scheme://example.com/path'),
    ))
def test_urldefragauth(url, expected):
    assert urldefragauth(url) == expected


@pytest.mark.parametrize(
    'url, expected', (
            ('http://192.168.0.1:5000/', True),
            ('http://192.168.0.1/', True),
            ('http://172.16.1.1/', True),
            ('http://172.16.1.1:5000/', True),
            ('http://localhost.localdomain:5000/v1.0/', True),
            ('http://172.16.1.12/', False),
            ('http://172.16.1.12:5000/', False),
            ('http://google.com:5000/v1.0/', False),
    ))
def test_should_bypass_proxies(url, expected, monkeypatch):
    """Tests for function should_bypass_proxies to check if proxy
    can be bypassed or not
    """
    monkeypatch.setenv('no_proxy', '192.168.0.0/24,127.0.0.1,localhost.localdomain,172.16.1.1')
    monkeypatch.setenv('NO_PROXY', '192.168.0.0/24,127.0.0.1,localhost.localdomain,172.16.1.1')
    assert should_bypass_proxies(url, no_proxy=None) == expected


@pytest.mark.parametrize(
    'cookiejar', (
        compat.cookielib.CookieJar(),
        RequestsCookieJar()
    ))
def test_add_dict_to_cookiejar(cookiejar):
    """Ensure add_dict_to_cookiejar works for
    non-RequestsCookieJar CookieJars
    """
    cookiedict = {'test': 'cookies',
                  'good': 'cookies'}
    cj = add_dict_to_cookiejar(cookiejar, cookiedict)
    cookies = dict((cookie.name, cookie.value) for cookie in cj)
    assert cookiedict == cookies


@pytest.mark.parametrize(
    'value, expected', (
                (u'test', True),
                (u'æíöû', False),
                (u'ジェーピーニック', False),
    )
)
def test_unicode_is_ascii(value, expected):
    assert unicode_is_ascii(value) is expected


@pytest.mark.parametrize(
    'url, expected', (
            ('http://192.168.0.1:5000/', True),
            ('http://192.168.0.1/', True),
            ('http://172.16.1.1/', True),
            ('http://172.16.1.1:5000/', True),
            ('http://localhost.localdomain:5000/v1.0/', True),
            ('http://172.16.1.12/', False),
            ('http://172.16.1.12:5000/', False),
            ('http://google.com:5000/v1.0/', False),
    ))
def test_should_bypass_proxies_no_proxy(
        url, expected, monkeypatch):
    """Tests for function should_bypass_proxies to check if proxy
    can be bypassed or not using the 'no_proxy' argument
    """
    no_proxy = '192.168.0.0/24,127.0.0.1,localhost.localdomain,172.16.1.1'
    # Test 'no_proxy' argument
    assert should_bypass_proxies(url, no_proxy=no_proxy) == expected


@pytest.mark.skipif(os.name != 'nt', reason='Test only on Windows')
@pytest.mark.parametrize(
    'url, expected, override', (
            ('http://192.168.0.1:5000/', True, None),
            ('http://192.168.0.1/', True, None),
            ('http://172.16.1.1/', True, None),
            ('http://172.16.1.1:5000/', True, None),
            ('http://localhost.localdomain:5000/v1.0/', True, None),
            ('http://172.16.1.22/', False, None),
            ('http://172.16.1.22:5000/', False, None),
            ('http://google.com:5000/v1.0/', False, None),
            ('http://mylocalhostname:5000/v1.0/', True, '<local>'),
            ('http://192.168.0.1/', False, ''),
    ))
def test_should_bypass_proxies_win_registry(url, expected, override,
                                            monkeypatch):
    """Tests for function should_bypass_proxies to check if proxy
    can be bypassed or not with Windows registry settings
    """
    if override is None:
        override = '192.168.*;127.0.0.1;localhost.localdomain;172.16.1.1'
    if compat.is_py3:
        import winreg
    else:
        import _winreg as winreg

    class RegHandle:
        def Close(self):
            pass

    ie_settings = RegHandle()

    def OpenKey(key, subkey):
        return ie_settings

    def QueryValueEx(key, value_name):
        if key is ie_settings:
            if value_name == 'ProxyEnable':
                return [1]
            elif value_name == 'ProxyOverride':
                return [override]

    monkeypatch.setenv('http_proxy', '')
    monkeypatch.setenv('https_proxy', '')
    monkeypatch.setenv('ftp_proxy', '')
    monkeypatch.setenv('no_proxy', '')
    monkeypatch.setenv('NO_PROXY', '')
    monkeypatch.setattr(winreg, 'OpenKey', OpenKey)
    monkeypatch.setattr(winreg, 'QueryValueEx', QueryValueEx)


@pytest.mark.parametrize(
    'env_name, value', (
            ('no_proxy', '192.168.0.0/24,127.0.0.1,localhost.localdomain'),
            ('no_proxy', None),
            ('a_new_key', '192.168.0.0/24,127.0.0.1,localhost.localdomain'),
            ('a_new_key', None),
    ))
def test_set_environ(env_name, value):
    """Tests set_environ will set environ values and will restore the environ."""
    environ_copy = copy.deepcopy(os.environ)
    with set_environ(env_name, value):
        assert os.environ.get(env_name) == value

    assert os.environ == environ_copy


def test_set_environ_raises_exception():
    """Tests set_environ will raise exceptions in context when the
    value parameter is None."""
    with pytest.raises(Exception) as exception:
        with set_environ('test1', None):
            raise Exception('Expected exception')

    assert 'Expected exception' in str(exception.value)
