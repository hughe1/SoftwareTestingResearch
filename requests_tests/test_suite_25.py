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


class TestGetEnvironProxies:
    """Ensures that IP addresses are correctly matches with ranges
    in no_proxy variable.
    """

    @pytest.fixture(autouse=True, params=['no_proxy', 'NO_PROXY'])
    def no_proxy(self, request, monkeypatch):
        monkeypatch.setenv(request.param, '192.168.0.0/24,127.0.0.1,localhost.localdomain,172.16.1.1')

    @pytest.mark.parametrize(
        'url', (
            'http://192.168.0.1:5000/',
            'http://192.168.0.1/',
            'http://172.16.1.1/',
            'http://172.16.1.1:5000/',
            'http://localhost.localdomain:5000/v1.0/',
        ))
    def test_bypass(self, url):
        assert get_environ_proxies(url, no_proxy=None) == {}

    @pytest.mark.parametrize(
        'url', (
            'http://192.168.1.1:5000/',
            'http://192.168.1.1/',
            'http://www.requests.com/',
        ))
    def test_not_bypass(self, url):
        assert get_environ_proxies(url, no_proxy=None) != {}

    @pytest.mark.parametrize(
        'url', (
            'http://192.168.1.1:5000/',
            'http://192.168.1.1/',
            'http://www.requests.com/',
        ))
    def test_bypass_no_proxy_keyword(self, url):
        no_proxy = '192.168.1.1,requests.com'
        assert get_environ_proxies(url, no_proxy=no_proxy) == {}

    @pytest.mark.parametrize(
        'url', (
            'http://192.168.0.1:5000/',
            'http://192.168.0.1/',
            'http://172.16.1.1/',
            'http://172.16.1.1:5000/',
            'http://localhost.localdomain:5000/v1.0/',
        ))
    def test_not_bypass_no_proxy_keyword(self, url, monkeypatch):
        # This is testing that the 'no_proxy' argument overrides the
        # environment variable 'no_proxy'
        monkeypatch.setenv('http_proxy', 'http://proxy.example.com:3128/')
        no_proxy = '192.168.1.1,requests.com'
        assert get_environ_proxies(url, no_proxy=no_proxy) != {}


def test_redirect_rfc1808_to_non_ascii_location():
    path = u'š'
    expected_path = b'%C5%A1'
    redirect_request = []  # stores the second request to the server

    def redirect_resp_handler(sock):
        consume_socket_content(sock, timeout=0.5)
        location = u'//{0}:{1}/{2}'.format(host, port, path)
        sock.send(
            b'HTTP/1.1 301 Moved Permanently\r\n'
            b'Content-Length: 0\r\n'
            b'Location: ' + location.encode('utf8') + b'\r\n'
            b'\r\n'
        )
        redirect_request.append(consume_socket_content(sock, timeout=0.5))
        sock.send(b'HTTP/1.1 200 OK\r\n\r\n')

    close_server = threading.Event()
    server = Server(redirect_resp_handler, wait_to_close_event=close_server)

    with server as (host, port):
        url = u'http://{0}:{1}'.format(host, port)
        r = requests.get(url=url, allow_redirects=True)
        assert r.status_code == 200
        assert len(r.history) == 1
        assert r.history[0].status_code == 301
        assert redirect_request[0].startswith(b'GET /' + expected_path + b' HTTP/1.1')
        assert r.url == u'{0}/{1}'.format(url, expected_path.decode('ascii'))

        close_server.set()

class TestMorselToCookieExpires:
    """Tests for morsel_to_cookie when morsel contains expires."""

    def test_expires_valid_str(self):
        """Test case where we convert expires from string time."""

        morsel = Morsel()
        morsel['expires'] = 'Thu, 01-Jan-1970 00:00:01 GMT'
        cookie = morsel_to_cookie(morsel)
        assert cookie.expires == 1

    @pytest.mark.parametrize(
        'value, exception', (
            (100, TypeError),
            ('woops', ValueError),
        ))
    def test_expires_invalid_int(self, value, exception):
        """Test case where an invalid type is passed for expires."""
        morsel = Morsel()
        morsel['expires'] = value
        with pytest.raises(exception):
            morsel_to_cookie(morsel)

    def test_expires_none(self):
        """Test case where expires is None."""

        morsel = Morsel()
        morsel['expires'] = None
        cookie = morsel_to_cookie(morsel)
        assert cookie.expires is None


class TestSuperLen:



    @pytest.mark.parametrize('error', [IOError, OSError])
    def test_super_len_handles_files_raising_weird_errors_in_tell(self, error):
        """If tell() raises errors, assume the cursor is at position zero."""
        class BoomFile(object):
            def __len__(self):
                return 5

            def tell(self):
                raise error()

        assert super_len(BoomFile()) == 0

    @pytest.mark.parametrize('error', [IOError, OSError])
    def test_super_len_tell_ioerror(self, error):
        """Ensure that if tell gives an IOError super_len doesn't fail"""
        class NoLenBoomFile(object):
            def tell(self):
                raise error()

            def seek(self, offset, whence):
                pass

        assert super_len(NoLenBoomFile()) == 0

    def test_string(self):
        assert super_len('Test') == 4

    @pytest.mark.parametrize(
        'mode, warnings_num', (
            ('r', 1),
            ('rb', 0),
        ))
    def test_file(self, tmpdir, mode, warnings_num, recwarn):
        file_obj = tmpdir.join('test.txt')
        file_obj.write('Test')
        with file_obj.open(mode) as fd:
            assert super_len(fd) == 4
        assert len(recwarn) == warnings_num

    def test_super_len_with__len__(self):
        foo = [1,2,3,4]
        len_foo = super_len(foo)
        assert len_foo == 4

    def test_super_len_with_no__len__(self):
        class LenFile(object):
            def __init__(self):
                self.len = 5

        assert super_len(LenFile()) == 5



    def test_super_len_with_fileno(self):
        with open(__file__, 'rb') as f:
            length = super_len(f)
            file_data = f.read()
        assert length == len(file_data)

    def test_super_len_with_no_matches(self):
        """Ensure that objects without any length methods default to 0"""
        assert super_len(object()) == 0


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



