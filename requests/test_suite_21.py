# -*- coding: utf-8 -*-

import threading
import socket
import time

import pytest
import requests
from tests.testserver.server import Server

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

from .compat import StringIO, cStringIO

import requests

# -*- coding: utf-8 -*-

import pytest

from requests import hooks

# -*- encoding: utf-8

import sys

import pytest

from requests.help import info

# -*- coding: utf-8 -*-

import pytest
import threading
import requests

from tests.testserver.server import Server, consume_socket_content

from .utils import override_environ

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

from .compat import StringIO, u
from .utils import override_environ
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


@pytest.mark.skipif(sys.version_info[:2] != (2,6), reason="Only run on Python 2.6")
def test_system_ssl_py26():
    """OPENSSL_VERSION_NUMBER isn't provided in Python 2.6, verify we don't
    blow up in this case.
    """
    assert info()['system_ssl'] == {'version': ''}


@pytest.mark.skipif(sys.version_info[:2] != (2,6), reason="Only run on Python 2.6")
def test_system_ssl_py26():
    """OPENSSL_VERSION_NUMBER isn't provided in Python 2.6, verify we don't
    blow up in this case.
    """
    assert info()['system_ssl'] == {'version': ''}


def test_chunked_upload():
    """can safely send generators"""
    close_server = threading.Event()
    server = Server.basic_response_server(wait_to_close_event=close_server)
    data = iter([b'a', b'b', b'c'])

    with server as (host, port):
        url = 'http://{0}:{1}/'.format(host, port)
        r = requests.post(url, data=data, stream=True)
        close_server.set()  # release server block

    assert r.status_code == 200
    assert r.request.headers['Transfer-Encoding'] == 'chunked'


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


@pytest.mark.parametrize(
    'hooks_list, result', (
        (hook, 'ata'),
        ([hook, lambda x: None, hook], 'ta'),
    )
)
def test_hooks(hooks_list, result):
    assert hooks.dispatch_hook('response', {'response': hooks_list}, 'Data') == result


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

_proxy_combos = []
for prefix, schemes in _schemes_by_var_prefix:
    for scheme in schemes:
        _proxy_combos.append(("{0}_proxy".format(prefix), scheme))

_proxy_combos += [(var.upper(), scheme) for var, scheme in _proxy_combos]


class TestIsIPv4Address:

    def test_valid(self):
        assert is_ipv4_address('8.8.8.8')

    @pytest.mark.parametrize('value', ('8.8.8.8.8', 'localhost.localdomain'))
    def test_invalid(self, value):
        assert not is_ipv4_address(value)


class TestRequests:

    def test_entry_points(self):

        requests.session
        requests.session().get
        requests.session().head
        requests.get
        requests.head
        requests.put
        requests.patch
        requests.post
        # Not really an entry point, but people rely on it.
        from requests.packages.urllib3.poolmanager import PoolManager

    @pytest.mark.parametrize(
        'exception, url', (
            (MissingSchema, 'hiwpefhipowhefopw'),
            (InvalidSchema, 'localhost:3128'),
            (InvalidSchema, 'localhost.localdomain:3128/'),
            (InvalidSchema, '10.122.1.1:3128/'),
            (InvalidURL, 'http://'),
        ))
    def test_invalid_url(self, exception, url):
        with pytest.raises(exception):
            requests.get(url)

    def test_basic_building(self):
        req = requests.Request()
        req.url = 'http://kennethreitz.org/'
        req.data = {'life': '42'}

        pr = req.prepare()
        assert pr.url == req.url
        assert pr.body == 'life=42'

    @pytest.mark.parametrize('method', ('GET', 'HEAD'))
    def test_no_content_length(self, httpbin, method):
        req = requests.Request(method, httpbin(method.lower())).prepare()
        assert 'Content-Length' not in req.headers

    @pytest.mark.parametrize('method', ('POST', 'PUT', 'PATCH', 'OPTIONS'))
    def test_no_body_content_length(self, httpbin, method):
        req = requests.Request(method, httpbin(method.lower())).prepare()
        assert req.headers['Content-Length'] == '0'

    @pytest.mark.parametrize('method', ('POST', 'PUT', 'PATCH', 'OPTIONS'))
    def test_empty_content_length(self, httpbin, method):
        req = requests.Request(method, httpbin(method.lower()), data='').prepare()
        assert req.headers['Content-Length'] == '0'

    def test_override_content_length(self, httpbin):
        headers = {
            'Content-Length': 'not zero'
        }
        r = requests.Request('POST', httpbin('post'), headers=headers).prepare()
        assert 'Content-Length' in r.headers
        assert r.headers['Content-Length'] == 'not zero'

    def test_path_is_not_double_encoded(self):
        request = requests.Request('GET', "http://0.0.0.0/get/test case").prepare()

        assert request.path_url == '/get/test%20case'

    @pytest.mark.parametrize(
        'url, expected', (
            ('http://example.com/path#fragment', 'http://example.com/path?a=b#fragment'),
            ('http://example.com/path?key=value#fragment', 'http://example.com/path?key=value&a=b#fragment')
        ))
    def test_params_are_added_before_fragment(self, url, expected):
        request = requests.Request('GET', url, params={"a": "b"}).prepare()
        assert request.url == expected

    def test_params_original_order_is_preserved_by_default(self):
        param_ordered_dict = OrderedDict((('z', 1), ('a', 1), ('k', 1), ('d', 1)))
        session = requests.Session()
        request = requests.Request('GET', 'http://example.com/', params=param_ordered_dict)
        prep = session.prepare_request(request)
        assert prep.url == 'http://example.com/?z=1&a=1&k=1&d=1'

    def test_params_bytes_are_encoded(self):
        request = requests.Request('GET', 'http://example.com',
                                   params=b'test=foo').prepare()
        assert request.url == 'http://example.com/?test=foo'

    def test_binary_put(self):
        request = requests.Request('PUT', 'http://example.com',
                                   data=u"ööö".encode("utf-8")).prepare()
        assert isinstance(request.body, bytes)

    def test_whitespaces_are_removed_from_url(self):
        # Test for issue #3696
        request = requests.Request('GET', ' http://example.com').prepare()
        assert request.url == 'http://example.com/'

    @pytest.mark.parametrize('scheme', ('http://', 'HTTP://', 'hTTp://', 'HttP://'))
    def test_mixed_case_scheme_acceptable(self, httpbin, scheme):
        s = requests.Session()
        s.proxies = getproxies()
        parts = urlparse(httpbin('get'))
        url = scheme + parts.netloc + parts.path
        r = requests.Request('GET', url)
        r = s.send(r.prepare())
        assert r.status_code == 200, 'failed for scheme {0}'.format(scheme)

    def test_HTTP_200_OK_GET_ALTERNATIVE(self, httpbin):
        r = requests.Request('GET', httpbin('get'))
        s = requests.Session()
        s.proxies = getproxies()

        r = s.send(r.prepare())

        assert r.status_code == 200

    def test_HTTP_302_ALLOW_REDIRECT_GET(self, httpbin):
        r = requests.get(httpbin('redirect', '1'))
        assert r.status_code == 200
        assert r.history[0].status_code == 302
        assert r.history[0].is_redirect

    def test_HTTP_307_ALLOW_REDIRECT_POST(self, httpbin):
        r = requests.post(httpbin('redirect-to'), data='test', params={'url': 'post', 'status_code': 307})
        assert r.status_code == 200
        assert r.history[0].status_code == 307
        assert r.history[0].is_redirect
        assert r.json()['data'] == 'test'

    def test_HTTP_307_ALLOW_REDIRECT_POST_WITH_SEEKABLE(self, httpbin):
        byte_str = b'test'
        r = requests.post(httpbin('redirect-to'), data=io.BytesIO(byte_str), params={'url': 'post', 'status_code': 307})
        assert r.status_code == 200
        assert r.history[0].status_code == 307
        assert r.history[0].is_redirect
        assert r.json()['data'] == byte_str.decode('utf-8')

    def test_HTTP_302_TOO_MANY_REDIRECTS(self, httpbin):
        try:
            requests.get(httpbin('relative-redirect', '50'))
        except TooManyRedirects as e:
            url = httpbin('relative-redirect', '20')
            assert e.request.url == url
            assert e.response.url == url
            assert len(e.response.history) == 30
        else:
            pytest.fail('Expected redirect to raise TooManyRedirects but it did not')

    def test_HTTP_302_TOO_MANY_REDIRECTS_WITH_PARAMS(self, httpbin):
        s = requests.session()
        s.max_redirects = 5
        try:
            s.get(httpbin('relative-redirect', '50'))
        except TooManyRedirects as e:
            url = httpbin('relative-redirect', '45')
            assert e.request.url == url
            assert e.response.url == url
            assert len(e.response.history) == 5
        else:
            pytest.fail('Expected custom max number of redirects to be respected but was not')

    def test_http_301_changes_post_to_get(self, httpbin):
        r = requests.post(httpbin('status', '301'))
        assert r.status_code == 200
        assert r.request.method == 'GET'
        assert r.history[0].status_code == 301
        assert r.history[0].is_redirect

    def test_http_301_doesnt_change_head_to_get(self, httpbin):
        r = requests.head(httpbin('status', '301'), allow_redirects=True)
        print(r.content)
        assert r.status_code == 200
        assert r.request.method == 'HEAD'
        assert r.history[0].status_code == 301
        assert r.history[0].is_redirect

    def test_http_302_changes_post_to_get(self, httpbin):
        r = requests.post(httpbin('status', '302'))
        assert r.status_code == 200
        assert r.request.method == 'GET'
        assert r.history[0].status_code == 302
        assert r.history[0].is_redirect

    def test_http_302_doesnt_change_head_to_get(self, httpbin):
        r = requests.head(httpbin('status', '302'), allow_redirects=True)
        assert r.status_code == 200
        assert r.request.method == 'HEAD'
        assert r.history[0].status_code == 302
        assert r.history[0].is_redirect

    def test_http_303_changes_post_to_get(self, httpbin):
        r = requests.post(httpbin('status', '303'))
        assert r.status_code == 200
        assert r.request.method == 'GET'
        assert r.history[0].status_code == 303
        assert r.history[0].is_redirect

    def test_http_303_doesnt_change_head_to_get(self, httpbin):
        r = requests.head(httpbin('status', '303'), allow_redirects=True)
        assert r.status_code == 200
        assert r.request.method == 'HEAD'
        assert r.history[0].status_code == 303
        assert r.history[0].is_redirect

    def test_header_and_body_removal_on_redirect(self, httpbin):
        purged_headers = ('Content-Length', 'Content-Type')
        ses = requests.Session()
        req = requests.Request('POST', httpbin('post'), data={'test': 'data'})
        prep = ses.prepare_request(req)
        resp = ses.send(prep)

        # Mimic a redirect response
        resp.status_code = 302
        resp.headers['location'] = 'get'

        # Run request through resolve_redirects
        next_resp = next(ses.resolve_redirects(resp, prep))
        assert next_resp.request.body is None
        for header in purged_headers:
            assert header not in next_resp.request.headers

    def test_transfer_enc_removal_on_redirect(self, httpbin):
        purged_headers = ('Transfer-Encoding', 'Content-Type')
        ses = requests.Session()
        req = requests.Request('POST', httpbin('post'), data=(b'x' for x in range(1)))
        prep = ses.prepare_request(req)
        assert 'Transfer-Encoding' in prep.headers

        # Create Response to avoid https://github.com/kevin1024/pytest-httpbin/issues/33
        resp = requests.Response()
        resp.raw = io.BytesIO(b'the content')
        resp.request = prep
        setattr(resp.raw, 'release_conn', lambda *args: args)

        # Mimic a redirect response
        resp.status_code = 302
        resp.headers['location'] = httpbin('get')

        # Run request through resolve_redirect
        next_resp = next(ses.resolve_redirects(resp, prep))
        assert next_resp.request.body is None
        for header in purged_headers:
            assert header not in next_resp.request.headers

    def test_HTTP_200_OK_GET_WITH_PARAMS(self, httpbin):
        heads = {'User-agent': 'Mozilla/5.0'}

        r = requests.get(httpbin('user-agent'), headers=heads)

        assert heads['User-agent'] in r.text
        assert r.status_code == 200

    def test_HTTP_200_OK_GET_WITH_MIXED_PARAMS(self, httpbin):
        heads = {'User-agent': 'Mozilla/5.0'}

        r = requests.get(httpbin('get') + '?test=true', params={'q': 'test'}, headers=heads)
        assert r.status_code == 200

    def test_set_cookie_on_301(self, httpbin):
        s = requests.session()
        url = httpbin('cookies/set?foo=bar')
        s.get(url)
        assert s.cookies['foo'] == 'bar'

    def test_cookie_sent_on_redirect(self, httpbin):
        s = requests.session()
        s.get(httpbin('cookies/set?foo=bar'))
        r = s.get(httpbin('redirect/1'))  # redirects to httpbin('get')
        assert 'Cookie' in r.json()['headers']

    def test_cookie_removed_on_expire(self, httpbin):
        s = requests.session()
        s.get(httpbin('cookies/set?foo=bar'))
        assert s.cookies['foo'] == 'bar'
        s.get(
            httpbin('response-headers'),
            params={
                'Set-Cookie':
                    'foo=deleted; expires=Thu, 01-Jan-1970 00:00:01 GMT'
            }
        )
        assert 'foo' not in s.cookies

    def test_cookie_quote_wrapped(self, httpbin):
        s = requests.session()
        s.get(httpbin('cookies/set?foo="bar:baz"'))
        assert s.cookies['foo'] == '"bar:baz"'

    def test_cookie_persists_via_api(self, httpbin):
        s = requests.session()
        r = s.get(httpbin('redirect/1'), cookies={'foo': 'bar'})
        assert 'foo' in r.request.headers['Cookie']
        assert 'foo' in r.history[0].request.headers['Cookie']

    def test_request_cookie_overrides_session_cookie(self, httpbin):
        s = requests.session()
        s.cookies['foo'] = 'bar'
        r = s.get(httpbin('cookies'), cookies={'foo': 'baz'})
        assert r.json()['cookies']['foo'] == 'baz'
        # Session cookie should not be modified
        assert s.cookies['foo'] == 'bar'

    def test_request_cookies_not_persisted(self, httpbin):
        s = requests.session()
        s.get(httpbin('cookies'), cookies={'foo': 'baz'})
        # Sending a request with cookies should not add cookies to the session
        assert not s.cookies

    def test_generic_cookiejar_works(self, httpbin):
        cj = cookielib.CookieJar()
        cookiejar_from_dict({'foo': 'bar'}, cj)
        s = requests.session()
        s.cookies = cj
        r = s.get(httpbin('cookies'))
        # Make sure the cookie was sent
        assert r.json()['cookies']['foo'] == 'bar'
        # Make sure the session cj is still the custom one
        assert s.cookies is cj

    def test_param_cookiejar_works(self, httpbin):
        cj = cookielib.CookieJar()
        cookiejar_from_dict({'foo': 'bar'}, cj)
        s = requests.session()
        r = s.get(httpbin('cookies'), cookies=cj)
        # Make sure the cookie was sent
        assert r.json()['cookies']['foo'] == 'bar'

    def test_cookielib_cookiejar_on_redirect(self, httpbin):
        """Tests resolve_redirect doesn't fail when merging cookies
        with non-RequestsCookieJar cookiejar.

        See GH #3579
        """
        cj = cookiejar_from_dict({'foo': 'bar'}, cookielib.CookieJar())
        s = requests.Session()
        s.cookies = cookiejar_from_dict({'cookie': 'tasty'})

        # Prepare request without using Session
        req = requests.Request('GET', httpbin('headers'), cookies=cj)
        prep_req = req.prepare()

        # Send request and simulate redirect
        resp = s.send(prep_req)
        resp.status_code = 302
        resp.headers['location'] = httpbin('get')
        redirects = s.resolve_redirects(resp, prep_req)
        resp = next(redirects)

        # Verify CookieJar isn't being converted to RequestsCookieJar
        assert isinstance(prep_req._cookies, cookielib.CookieJar)
        assert isinstance(resp.request._cookies, cookielib.CookieJar)
        assert not isinstance(resp.request._cookies, requests.cookies.RequestsCookieJar)

        cookies = {}
        for c in resp.request._cookies:
            cookies[c.name] = c.value
        assert cookies['foo'] == 'bar'
        assert cookies['cookie'] == 'tasty'

    def test_requests_in_history_are_not_overridden(self, httpbin):
        resp = requests.get(httpbin('redirect/3'))
        urls = [r.url for r in resp.history]
        req_urls = [r.request.url for r in resp.history]
        assert urls == req_urls

    def test_history_is_always_a_list(self, httpbin):
        """Show that even with redirects, Response.history is always a list."""
        resp = requests.get(httpbin('get'))
        assert isinstance(resp.history, list)
        resp = requests.get(httpbin('redirect/1'))
        assert isinstance(resp.history, list)
        assert not isinstance(resp.history, tuple)

    def test_headers_on_session_with_None_are_not_sent(self, httpbin):
        """Do not send headers in Session.headers with None values."""
        ses = requests.Session()
        ses.headers['Accept-Encoding'] = None
        req = requests.Request('GET', httpbin('get'))
        prep = ses.prepare_request(req)
        assert 'Accept-Encoding' not in prep.headers

    def test_headers_preserve_order(self, httpbin):
        """Preserve order when headers provided as OrderedDict."""
        ses = requests.Session()
        ses.headers = OrderedDict()
        ses.headers['Accept-Encoding'] = 'identity'
        ses.headers['First'] = '1'
        ses.headers['Second'] = '2'
        headers = OrderedDict([('Third', '3'), ('Fourth', '4')])
        headers['Fifth'] = '5'
        headers['Second'] = '222'
        req = requests.Request('GET', httpbin('get'), headers=headers)
        prep = ses.prepare_request(req)
        items = list(prep.headers.items())
        assert items[0] == ('Accept-Encoding', 'identity')
        assert items[1] == ('First', '1')
        assert items[2] == ('Second', '222')
        assert items[3] == ('Third', '3')
        assert items[4] == ('Fourth', '4')
        assert items[5] == ('Fifth', '5')

    @pytest.mark.parametrize('key', ('User-agent', 'user-agent'))
    def test_user_agent_transfers(self, httpbin, key):

        heads = {key: 'Mozilla/5.0 (github.com/requests/requests)'}

        r = requests.get(httpbin('user-agent'), headers=heads)
        assert heads[key] in r.text

    def test_HTTP_200_OK_HEAD(self, httpbin):
        r = requests.head(httpbin('get'))
        assert r.status_code == 200

    def test_HTTP_200_OK_PUT(self, httpbin):
        r = requests.put(httpbin('put'))
        assert r.status_code == 200

    def test_BASICAUTH_TUPLE_HTTP_200_OK_GET(self, httpbin):
        auth = ('user', 'pass')
        url = httpbin('basic-auth', 'user', 'pass')

        r = requests.get(url, auth=auth)
        assert r.status_code == 200

        r = requests.get(url)
        assert r.status_code == 401

        s = requests.session()
        s.auth = auth
        r = s.get(url)
        assert r.status_code == 200

    @pytest.mark.parametrize(
        'username, password', (
            ('user', 'pass'),
            (u'имя'.encode('utf-8'), u'пароль'.encode('utf-8')),
            (42, 42),
            (None, None),
        ))
    def test_set_basicauth(self, httpbin, username, password):
        auth = (username, password)
        url = httpbin('get')

        r = requests.Request('GET', url, auth=auth)
        p = r.prepare()

        assert p.headers['Authorization'] == _basic_auth_str(username, password)

    def test_basicauth_encodes_byte_strings(self):
        """Ensure b'test' formats as the byte string "test" rather
        than the unicode string "b'test'" in Python 3.
        """
        auth = (b'\xc5\xafsername', b'test\xc6\xb6')
        r = requests.Request('GET', 'http://localhost', auth=auth)
        p = r.prepare()

        assert p.headers['Authorization'] == 'Basic xa9zZXJuYW1lOnRlc3TGtg=='

    @pytest.mark.parametrize(
        'url, exception', (
            # Connecting to an unknown domain should raise a ConnectionError
            ('http://doesnotexist.google.com', ConnectionError),
            # Connecting to an invalid port should raise a ConnectionError
            ('http://localhost:1', ConnectionError),
            # Inputing a URL that cannot be parsed should raise an InvalidURL error
            ('http://fe80::5054:ff:fe5a:fc0', InvalidURL)
        ))
    def test_errors(self, url, exception):
        with pytest.raises(exception):
            requests.get(url, timeout=1)

    def test_proxy_error(self):
        # any proxy related error (address resolution, no route to host, etc) should result in a ProxyError
        with pytest.raises(ProxyError):
            requests.get('http://localhost:1', proxies={'http': 'non-resolvable-address'})

    def test_basicauth_with_netrc(self, httpbin):
        auth = ('user', 'pass')
        wrong_auth = ('wronguser', 'wrongpass')
        url = httpbin('basic-auth', 'user', 'pass')

        old_auth = requests.sessions.get_netrc_auth

        try:
            def get_netrc_auth_mock(url):
                return auth
            requests.sessions.get_netrc_auth = get_netrc_auth_mock

            # Should use netrc and work.
            r = requests.get(url)
            assert r.status_code == 200

            # Given auth should override and fail.
            r = requests.get(url, auth=wrong_auth)
            assert r.status_code == 401

            s = requests.session()

            # Should use netrc and work.
            r = s.get(url)
            assert r.status_code == 200

            # Given auth should override and fail.
            s.auth = wrong_auth
            r = s.get(url)
            assert r.status_code == 401
        finally:
            requests.sessions.get_netrc_auth = old_auth

    def test_DIGEST_HTTP_200_OK_GET(self, httpbin):

        auth = HTTPDigestAuth('user', 'pass')
        url = httpbin('digest-auth', 'auth', 'user', 'pass')

        r = requests.get(url, auth=auth)
        assert r.status_code == 200

        r = requests.get(url)
        assert r.status_code == 401

        s = requests.session()
        s.auth = HTTPDigestAuth('user', 'pass')
        r = s.get(url)
        assert r.status_code == 200

    def test_DIGEST_AUTH_RETURNS_COOKIE(self, httpbin):
        url = httpbin('digest-auth', 'auth', 'user', 'pass')
        auth = HTTPDigestAuth('user', 'pass')
        r = requests.get(url)
        assert r.cookies['fake'] == 'fake_value'

        r = requests.get(url, auth=auth)
        assert r.status_code == 200

    def test_DIGEST_AUTH_SETS_SESSION_COOKIES(self, httpbin):
        url = httpbin('digest-auth', 'auth', 'user', 'pass')
        auth = HTTPDigestAuth('user', 'pass')
        s = requests.Session()
        s.get(url, auth=auth)
        assert s.cookies['fake'] == 'fake_value'

    def test_DIGEST_STREAM(self, httpbin):

        auth = HTTPDigestAuth('user', 'pass')
        url = httpbin('digest-auth', 'auth', 'user', 'pass')

        r = requests.get(url, auth=auth, stream=True)
        assert r.raw.read() != b''

        r = requests.get(url, auth=auth, stream=False)
        assert r.raw.read() == b''

    def test_DIGESTAUTH_WRONG_HTTP_401_GET(self, httpbin):

        auth = HTTPDigestAuth('user', 'wrongpass')
        url = httpbin('digest-auth', 'auth', 'user', 'pass')

        r = requests.get(url, auth=auth)
        assert r.status_code == 401

        r = requests.get(url)
        assert r.status_code == 401

        s = requests.session()
        s.auth = auth
        r = s.get(url)
        assert r.status_code == 401

    def test_DIGESTAUTH_QUOTES_QOP_VALUE(self, httpbin):

        auth = HTTPDigestAuth('user', 'pass')
        url = httpbin('digest-auth', 'auth', 'user', 'pass')

        r = requests.get(url, auth=auth)
        assert '"auth"' in r.request.headers['Authorization']

    def test_POSTBIN_GET_POST_FILES(self, httpbin):

        url = httpbin('post')
        requests.post(url).raise_for_status()

        post1 = requests.post(url, data={'some': 'data'})
        assert post1.status_code == 200

        with open('Pipfile') as f:
            post2 = requests.post(url, files={'some': f})
        assert post2.status_code == 200

        post4 = requests.post(url, data='[{"some": "json"}]')
        assert post4.status_code == 200

        with pytest.raises(ValueError):
            requests.post(url, files=['bad file data'])

    def test_POSTBIN_SEEKED_OBJECT_WITH_NO_ITER(self, httpbin):

        class TestStream(object):
            def __init__(self, data):
                self.data = data.encode()
                self.length = len(self.data)
                self.index = 0

            def __len__(self):
                return self.length

            def read(self, size=None):
                if size:
                    ret = self.data[self.index:self.index + size]
                    self.index += size
                else:
                    ret = self.data[self.index:]
                    self.index = self.length
                return ret

            def tell(self):
                return self.index

            def seek(self, offset, where=0):
                if where == 0:
                    self.index = offset
                elif where == 1:
                    self.index += offset
                elif where == 2:
                    self.index = self.length + offset

        test = TestStream('test')
        post1 = requests.post(httpbin('post'), data=test)
        assert post1.status_code == 200
        assert post1.json()['data'] == 'test'

        test = TestStream('test')
        test.seek(2)
        post2 = requests.post(httpbin('post'), data=test)
        assert post2.status_code == 200
        assert post2.json()['data'] == 'st'

    def test_POSTBIN_GET_POST_FILES_WITH_DATA(self, httpbin):

        url = httpbin('post')
        requests.post(url).raise_for_status()

        post1 = requests.post(url, data={'some': 'data'})
        assert post1.status_code == 200

        with open('Pipfile') as f:
            post2 = requests.post(url, data={'some': 'data'}, files={'some': f})
        assert post2.status_code == 200

        post4 = requests.post(url, data='[{"some": "json"}]')
        assert post4.status_code == 200

        with pytest.raises(ValueError):
            requests.post(url, files=['bad file data'])

    def test_post_with_custom_mapping(self, httpbin):
        class CustomMapping(collections.MutableMapping):
            def __init__(self, *args, **kwargs):
                self.data = dict(*args, **kwargs)

            def __delitem__(self, key):
                del self.data[key]

            def __getitem__(self, key):
                return self.data[key]

            def __setitem__(self, key, value):
                self.data[key] = value

            def __iter__(self):
                return iter(self.data)

            def __len__(self):
                return len(self.data)

        data = CustomMapping({'some': 'data'})
        url = httpbin('post')
        found_json = requests.post(url, data=data).json().get('form')
        assert found_json == {'some': 'data'}

    def test_conflicting_post_params(self, httpbin):
        url = httpbin('post')
        with open('Pipfile') as f:
            pytest.raises(ValueError, "requests.post(url, data='[{\"some\": \"data\"}]', files={'some': f})")
            pytest.raises(ValueError, "requests.post(url, data=u('[{\"some\": \"data\"}]'), files={'some': f})")

    def test_request_ok_set(self, httpbin):
        r = requests.get(httpbin('status', '404'))
        assert not r.ok

    def test_status_raising(self, httpbin):
        r = requests.get(httpbin('status', '404'))
        with pytest.raises(requests.exceptions.HTTPError):
            r.raise_for_status()

        r = requests.get(httpbin('status', '500'))
        assert not r.ok

    def test_decompress_gzip(self, httpbin):
        r = requests.get(httpbin('gzip'))
        r.content.decode('ascii')

    @pytest.mark.parametrize(
        'url, params', (
            ('/get', {'foo': 'føø'}),
            ('/get', {'føø': 'føø'}),
            ('/get', {'føø': 'føø'}),
            ('/get', {'foo': 'foo'}),
            ('ø', {'foo': 'foo'}),
        ))
    def test_unicode_get(self, httpbin, url, params):
        requests.get(httpbin(url), params=params)

    def test_unicode_header_name(self, httpbin):
        requests.put(
            httpbin('put'),
            headers={str('Content-Type'): 'application/octet-stream'},
            data='\xff')  # compat.str is unicode.

    def test_pyopenssl_redirect(self, httpbin_secure, httpbin_ca_bundle):
        requests.get(httpbin_secure('status', '301'), verify=httpbin_ca_bundle)

    def test_invalid_ca_certificate_path(self, httpbin_secure):
        INVALID_PATH = '/garbage'
        with pytest.raises(IOError) as e:
            requests.get(httpbin_secure(), verify=INVALID_PATH)
        assert str(e.value) == 'Could not find a suitable TLS CA certificate bundle, invalid path: {0}'.format(INVALID_PATH)

    def test_invalid_ssl_certificate_files(self, httpbin_secure):
        INVALID_PATH = '/garbage'
        with pytest.raises(IOError) as e:
            requests.get(httpbin_secure(), cert=INVALID_PATH)
        assert str(e.value) == 'Could not find the TLS certificate file, invalid path: {0}'.format(INVALID_PATH)

        with pytest.raises(IOError) as e:
            requests.get(httpbin_secure(), cert=('.', INVALID_PATH))
        assert str(e.value) == 'Could not find the TLS key file, invalid path: {0}'.format(INVALID_PATH)

    def test_http_with_certificate(self, httpbin):
        r = requests.get(httpbin(), cert='.')
        assert r.status_code == 200

    def test_https_warnings(self, httpbin_secure, httpbin_ca_bundle):
        """warnings are emitted with requests.get"""
        if HAS_MODERN_SSL or HAS_PYOPENSSL:
            warnings_expected = ('SubjectAltNameWarning', )
        else:
            warnings_expected = ('SNIMissingWarning',
                                 'InsecurePlatformWarning',
                                 'SubjectAltNameWarning', )

        with pytest.warns(None) as warning_records:
            warnings.simplefilter('always')
            requests.get(httpbin_secure('status', '200'),
                         verify=httpbin_ca_bundle)

        warning_records = [item for item in warning_records
                           if item.category.__name__ != 'ResourceWarning']

        warnings_category = tuple(
            item.category.__name__ for item in warning_records)
        assert warnings_category == warnings_expected

    def test_certificate_failure(self, httpbin_secure):
        """
        When underlying SSL problems occur, an SSLError is raised.
        """
        with pytest.raises(SSLError):
            # Our local httpbin does not have a trusted CA, so this call will
            # fail if we use our default trust bundle.
            requests.get(httpbin_secure('status', '200'))

    def test_urlencoded_get_query_multivalued_param(self, httpbin):

        r = requests.get(httpbin('get'), params=dict(test=['foo', 'baz']))
        assert r.status_code == 200
        assert r.url == httpbin('get?test=foo&test=baz')

    def test_different_encodings_dont_break_post(self, httpbin):
        r = requests.post(httpbin('post'),
            data={'stuff': json.dumps({'a': 123})},
            params={'blah': 'asdf1234'},
            files={'file': ('test_requests.py', open(__file__, 'rb'))})
        assert r.status_code == 200

    @pytest.mark.parametrize(
        'data', (
            {'stuff': u('ëlïxr')},
            {'stuff': u('ëlïxr').encode('utf-8')},
            {'stuff': 'elixr'},
            {'stuff': 'elixr'.encode('utf-8')},
        ))
    def test_unicode_multipart_post(self, httpbin, data):
        r = requests.post(httpbin('post'),
            data=data,
            files={'file': ('test_requests.py', open(__file__, 'rb'))})
        assert r.status_code == 200

    def test_unicode_multipart_post_fieldnames(self, httpbin):
        filename = os.path.splitext(__file__)[0] + '.py'
        r = requests.Request(
            method='POST', url=httpbin('post'),
            data={'stuff'.encode('utf-8'): 'elixr'},
            files={'file': ('test_requests.py', open(filename, 'rb'))})
        prep = r.prepare()
        assert b'name="stuff"' in prep.body
        assert b'name="b\'stuff\'"' not in prep.body

    def test_unicode_method_name(self, httpbin):
        files = {'file': open(__file__, 'rb')}
        r = requests.request(
            method=u('POST'), url=httpbin('post'), files=files)
        assert r.status_code == 200

    def test_unicode_method_name_with_request_object(self, httpbin):
        files = {'file': open(__file__, 'rb')}
        s = requests.Session()
        req = requests.Request(u('POST'), httpbin('post'), files=files)
        prep = s.prepare_request(req)
        assert isinstance(prep.method, builtin_str)
        assert prep.method == 'POST'

        resp = s.send(prep)
        assert resp.status_code == 200

    def test_non_prepared_request_error(self):
        s = requests.Session()
        req = requests.Request(u('POST'), '/')

        with pytest.raises(ValueError) as e:
            s.send(req)
        assert str(e.value) == 'You can only send PreparedRequests.'

    def test_custom_content_type(self, httpbin):
        r = requests.post(
            httpbin('post'),
            data={'stuff': json.dumps({'a': 123})},
            files={
                'file1': ('test_requests.py', open(__file__, 'rb')),
                'file2': ('test_requests', open(__file__, 'rb'),
                    'text/py-content-type')})
        assert r.status_code == 200
        assert b"text/py-content-type" in r.request.body

    def test_hook_receives_request_arguments(self, httpbin):
        def hook(resp, **kwargs):
            assert resp is not None
            assert kwargs != {}

        s = requests.Session()
        r = requests.Request('GET', httpbin(), hooks={'response': hook})
        prep = s.prepare_request(r)
        s.send(prep)

    def test_session_hooks_are_used_with_no_request_hooks(self, httpbin):
        hook = lambda x, *args, **kwargs: x
        s = requests.Session()
        s.hooks['response'].append(hook)
        r = requests.Request('GET', httpbin())
        prep = s.prepare_request(r)
        assert prep.hooks['response'] != []
        assert prep.hooks['response'] == [hook]

    def test_session_hooks_are_overridden_by_request_hooks(self, httpbin):
        hook1 = lambda x, *args, **kwargs: x
        hook2 = lambda x, *args, **kwargs: x
        assert hook1 is not hook2
        s = requests.Session()
        s.hooks['response'].append(hook2)
        r = requests.Request('GET', httpbin(), hooks={'response': [hook1]})
        prep = s.prepare_request(r)
        assert prep.hooks['response'] == [hook1]

    def test_prepared_request_hook(self, httpbin):
        def hook(resp, **kwargs):
            resp.hook_working = True
            return resp

        req = requests.Request('GET', httpbin(), hooks={'response': hook})
        prep = req.prepare()

        s = requests.Session()
        s.proxies = getproxies()
        resp = s.send(prep)

        assert hasattr(resp, 'hook_working')

    def test_prepared_from_session(self, httpbin):
        class DummyAuth(requests.auth.AuthBase):
            def __call__(self, r):
                r.headers['Dummy-Auth-Test'] = 'dummy-auth-test-ok'
                return r

        req = requests.Request('GET', httpbin('headers'))
        assert not req.auth

        s = requests.Session()
        s.auth = DummyAuth()

        prep = s.prepare_request(req)
        resp = s.send(prep)

        assert resp.json()['headers'][
            'Dummy-Auth-Test'] == 'dummy-auth-test-ok'

    def test_prepare_request_with_bytestring_url(self):
        req = requests.Request('GET', b'https://httpbin.org/')
        s = requests.Session()
        prep = s.prepare_request(req)
        assert prep.url == "https://httpbin.org/"

    def test_request_with_bytestring_host(self, httpbin):
        s = requests.Session()
        resp = s.request(
            'GET',
            httpbin('cookies/set?cookie=value'),
            allow_redirects=False,
            headers={'Host': b'httpbin.org'}
        )
        assert resp.cookies.get('cookie') == 'value'

    def test_links(self):
        r = requests.Response()
        r.headers = {
            'cache-control': 'public, max-age=60, s-maxage=60',
            'connection': 'keep-alive',
            'content-encoding': 'gzip',
            'content-type': 'application/json; charset=utf-8',
            'date': 'Sat, 26 Jan 2013 16:47:56 GMT',
            'etag': '"6ff6a73c0e446c1f61614769e3ceb778"',
            'last-modified': 'Sat, 26 Jan 2013 16:22:39 GMT',
            'link': ('<https://api.github.com/users/kennethreitz/repos?'
                     'page=2&per_page=10>; rel="next", <https://api.github.'
                     'com/users/kennethreitz/repos?page=7&per_page=10>; '
                     ' rel="last"'),
            'server': 'GitHub.com',
            'status': '200 OK',
            'vary': 'Accept',
            'x-content-type-options': 'nosniff',
            'x-github-media-type': 'github.beta',
            'x-ratelimit-limit': '60',
            'x-ratelimit-remaining': '57'
        }
        assert r.links['next']['rel'] == 'next'

    def test_cookie_parameters(self):
        key = 'some_cookie'
        value = 'some_value'
        secure = True
        domain = 'test.com'
        rest = {'HttpOnly': True}

        jar = requests.cookies.RequestsCookieJar()
        jar.set(key, value, secure=secure, domain=domain, rest=rest)

        assert len(jar) == 1
        assert 'some_cookie' in jar

        cookie = list(jar)[0]
        assert cookie.secure == secure
        assert cookie.domain == domain
        assert cookie._rest['HttpOnly'] == rest['HttpOnly']

    def test_cookie_as_dict_keeps_len(self):
        key = 'some_cookie'
        value = 'some_value'

        key1 = 'some_cookie1'
        value1 = 'some_value1'

        jar = requests.cookies.RequestsCookieJar()
        jar.set(key, value)
        jar.set(key1, value1)

        d1 = dict(jar)
        d2 = dict(jar.iteritems())
        d3 = dict(jar.items())

        assert len(jar) == 2
        assert len(d1) == 2
        assert len(d2) == 2
        assert len(d3) == 2

    def test_cookie_as_dict_keeps_items(self):
        key = 'some_cookie'
        value = 'some_value'

        key1 = 'some_cookie1'
        value1 = 'some_value1'

        jar = requests.cookies.RequestsCookieJar()
        jar.set(key, value)
        jar.set(key1, value1)

        d1 = dict(jar)
        d2 = dict(jar.iteritems())
        d3 = dict(jar.items())

        assert d1['some_cookie'] == 'some_value'
        assert d2['some_cookie'] == 'some_value'
        assert d3['some_cookie1'] == 'some_value1'

    def test_cookie_as_dict_keys(self):
        key = 'some_cookie'
        value = 'some_value'

        key1 = 'some_cookie1'
        value1 = 'some_value1'

        jar = requests.cookies.RequestsCookieJar()
        jar.set(key, value)
        jar.set(key1, value1)

        keys = jar.keys()
        assert keys == list(keys)
        # make sure one can use keys multiple times
        assert list(keys) == list(keys)

    def test_cookie_as_dict_values(self):
        key = 'some_cookie'
        value = 'some_value'

        key1 = 'some_cookie1'
        value1 = 'some_value1'

        jar = requests.cookies.RequestsCookieJar()
        jar.set(key, value)
        jar.set(key1, value1)

        values = jar.values()
        assert values == list(values)
        # make sure one can use values multiple times
        assert list(values) == list(values)

    def test_cookie_as_dict_items(self):
        key = 'some_cookie'
        value = 'some_value'

        key1 = 'some_cookie1'
        value1 = 'some_value1'

        jar = requests.cookies.RequestsCookieJar()
        jar.set(key, value)
        jar.set(key1, value1)

        items = jar.items()
        assert items == list(items)
        # make sure one can use items multiple times
        assert list(items) == list(items)

    def test_cookie_duplicate_names_different_domains(self):
        key = 'some_cookie'
        value = 'some_value'
        domain1 = 'test1.com'
        domain2 = 'test2.com'

        jar = requests.cookies.RequestsCookieJar()
        jar.set(key, value, domain=domain1)
        jar.set(key, value, domain=domain2)
        assert key in jar
        items = jar.items()
        assert len(items) == 2

        # Verify that CookieConflictError is raised if domain is not specified
        with pytest.raises(requests.cookies.CookieConflictError):
            jar.get(key)

        # Verify that CookieConflictError is not raised if domain is specified
        cookie = jar.get(key, domain=domain1)
        assert cookie == value

    def test_cookie_duplicate_names_raises_cookie_conflict_error(self):
        key = 'some_cookie'
        value = 'some_value'
        path = 'some_path'

        jar = requests.cookies.RequestsCookieJar()
        jar.set(key, value, path=path)
        jar.set(key, value)
        with pytest.raises(requests.cookies.CookieConflictError):
            jar.get(key)

    def test_time_elapsed_blank(self, httpbin):
        r = requests.get(httpbin('get'))
        td = r.elapsed
        total_seconds = ((td.microseconds + (td.seconds + td.days * 24 * 3600) * 10**6) / 10**6)
        assert total_seconds > 0.0

    def test_empty_response_has_content_none(self):
        r = requests.Response()
        assert r.content is None

    def test_response_is_iterable(self):
        r = requests.Response()
        io = StringIO.StringIO('abc')
        read_ = io.read

        def read_mock(amt, decode_content=None):
            return read_(amt)
        setattr(io, 'read', read_mock)
        r.raw = io
        assert next(iter(r))
        io.close()

    def test_response_decode_unicode(self):
        """When called with decode_unicode, Response.iter_content should always
        return unicode.
        """
        r = requests.Response()
        r._content_consumed = True
        r._content = b'the content'
        r.encoding = 'ascii'

        chunks = r.iter_content(decode_unicode=True)
        assert all(isinstance(chunk, str) for chunk in chunks)

        # also for streaming
        r = requests.Response()
        r.raw = io.BytesIO(b'the content')
        r.encoding = 'ascii'
        chunks = r.iter_content(decode_unicode=True)
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_response_reason_unicode(self):
        # check for unicode HTTP status
        r = requests.Response()
        r.url = u'unicode URL'
        r.reason = u'Komponenttia ei löydy'.encode('utf-8')
        r.status_code = 404
        r.encoding = None
        assert not r.ok  # old behaviour - crashes here

    def test_response_reason_unicode_fallback(self):
        # check raise_status falls back to ISO-8859-1
        r = requests.Response()
        r.url = 'some url'
        reason = u'Komponenttia ei löydy'
        r.reason = reason.encode('latin-1')
        r.status_code = 500
        r.encoding = None
        with pytest.raises(requests.exceptions.HTTPError) as e:
            r.raise_for_status()
        assert reason in e.value.args[0]

    def test_response_chunk_size_type(self):
        """Ensure that chunk_size is passed as None or an integer, otherwise
        raise a TypeError.
        """
        r = requests.Response()
        r.raw = io.BytesIO(b'the content')
        chunks = r.iter_content(1)
        assert all(len(chunk) == 1 for chunk in chunks)

        r = requests.Response()
        r.raw = io.BytesIO(b'the content')
        chunks = r.iter_content(None)
        assert list(chunks) == [b'the content']

        r = requests.Response()
        r.raw = io.BytesIO(b'the content')
        with pytest.raises(TypeError):
            chunks = r.iter_content("1024")

    def test_request_and_response_are_pickleable(self, httpbin):
        r = requests.get(httpbin('get'))

        # verify we can pickle the original request
        assert pickle.loads(pickle.dumps(r.request))

        # verify we can pickle the response and that we have access to
        # the original request.
        pr = pickle.loads(pickle.dumps(r))
        assert r.request.url == pr.request.url
        assert r.request.headers == pr.request.headers

    def test_prepared_request_is_pickleable(self, httpbin):
        p = requests.Request('GET', httpbin('get')).prepare()

        # Verify PreparedRequest can be pickled and unpickled
        r = pickle.loads(pickle.dumps(p))
        assert r.url == p.url
        assert r.headers == p.headers
        assert r.body == p.body

        # Verify unpickled PreparedRequest sends properly
        s = requests.Session()
        resp = s.send(r)
        assert resp.status_code == 200

    def test_prepared_request_with_file_is_pickleable(self, httpbin):
        files = {'file': open(__file__, 'rb')}
        r = requests.Request('POST', httpbin('post'), files=files)
        p = r.prepare()

        # Verify PreparedRequest can be pickled and unpickled
        r = pickle.loads(pickle.dumps(p))
        assert r.url == p.url
        assert r.headers == p.headers
        assert r.body == p.body

        # Verify unpickled PreparedRequest sends properly
        s = requests.Session()
        resp = s.send(r)
        assert resp.status_code == 200

    def test_prepared_request_with_hook_is_pickleable(self, httpbin):
        r = requests.Request('GET', httpbin('get'), hooks=default_hooks())
        p = r.prepare()

        # Verify PreparedRequest can be pickled
        r = pickle.loads(pickle.dumps(p))
        assert r.url == p.url
        assert r.headers == p.headers
        assert r.body == p.body
        assert r.hooks == p.hooks

        # Verify unpickled PreparedRequest sends properly
        s = requests.Session()
        resp = s.send(r)
        assert resp.status_code == 200

    def test_cannot_send_unprepared_requests(self, httpbin):
        r = requests.Request(url=httpbin())
        with pytest.raises(ValueError):
            requests.Session().send(r)

    def test_http_error(self):
        error = requests.exceptions.HTTPError()
        assert not error.response
        response = requests.Response()
        error = requests.exceptions.HTTPError(response=response)
        assert error.response == response
        error = requests.exceptions.HTTPError('message', response=response)
        assert str(error) == 'message'
        assert error.response == response

    def test_session_pickling(self, httpbin):
        r = requests.Request('GET', httpbin('get'))
        s = requests.Session()

        s = pickle.loads(pickle.dumps(s))
        s.proxies = getproxies()

        r = s.send(r.prepare())
        assert r.status_code == 200

    def test_fixes_1329(self, httpbin):
        """Ensure that header updates are done case-insensitively."""
        s = requests.Session()
        s.headers.update({'ACCEPT': 'BOGUS'})
        s.headers.update({'accept': 'application/json'})
        r = s.get(httpbin('get'))
        headers = r.request.headers
        assert headers['accept'] == 'application/json'
        assert headers['Accept'] == 'application/json'
        assert headers['ACCEPT'] == 'application/json'

    def test_uppercase_scheme_redirect(self, httpbin):
        parts = urlparse(httpbin('html'))
        url = "HTTP://" + parts.netloc + parts.path
        r = requests.get(httpbin('redirect-to'), params={'url': url})
        assert r.status_code == 200
        assert r.url.lower() == url.lower()

    def test_transport_adapter_ordering(self):
        s = requests.Session()
        order = ['https://', 'http://']
        assert order == list(s.adapters)
        s.mount('http://git', HTTPAdapter())
        s.mount('http://github', HTTPAdapter())
        s.mount('http://github.com', HTTPAdapter())
        s.mount('http://github.com/about/', HTTPAdapter())
        order = [
            'http://github.com/about/',
            'http://github.com',
            'http://github',
            'http://git',
            'https://',
            'http://',
        ]
        assert order == list(s.adapters)
        s.mount('http://gittip', HTTPAdapter())
        s.mount('http://gittip.com', HTTPAdapter())
        s.mount('http://gittip.com/about/', HTTPAdapter())
        order = [
            'http://github.com/about/',
            'http://gittip.com/about/',
            'http://github.com',
            'http://gittip.com',
            'http://github',
            'http://gittip',
            'http://git',
            'https://',
            'http://',
        ]
        assert order == list(s.adapters)
        s2 = requests.Session()
        s2.adapters = {'http://': HTTPAdapter()}
        s2.mount('https://', HTTPAdapter())
        assert 'http://' in s2.adapters
        assert 'https://' in s2.adapters

    def test_header_remove_is_case_insensitive(self, httpbin):
        # From issue #1321
        s = requests.Session()
        s.headers['foo'] = 'bar'
        r = s.get(httpbin('get'), headers={'FOO': None})
        assert 'foo' not in r.request.headers

    def test_params_are_merged_case_sensitive(self, httpbin):
        s = requests.Session()
        s.params['foo'] = 'bar'
        r = s.get(httpbin('get'), params={'FOO': 'bar'})
        assert r.json()['args'] == {'foo': 'bar', 'FOO': 'bar'}

    def test_long_authinfo_in_url(self):
        url = 'http://{0}:{1}@{2}:9000/path?query#frag'.format(
            'E8A3BE87-9E3F-4620-8858-95478E385B5B',
            'EA770032-DA4D-4D84-8CE9-29C6D910BF1E',
            'exactly-------------sixty-----------three------------characters',
        )
        r = requests.Request('GET', url).prepare()
        assert r.url == url

    def test_header_keys_are_native(self, httpbin):
        headers = {u('unicode'): 'blah', 'byte'.encode('ascii'): 'blah'}
        r = requests.Request('GET', httpbin('get'), headers=headers)
        p = r.prepare()

        # This is testing that they are builtin strings. A bit weird, but there
        # we go.
        assert 'unicode' in p.headers.keys()
        assert 'byte' in p.headers.keys()

    def test_header_validation(self, httpbin):
        """Ensure prepare_headers regex isn't flagging valid header contents."""
        headers_ok = {'foo': 'bar baz qux',
                      'bar': u'fbbq'.encode('utf8'),
                      'baz': '',
                      'qux': '1'}
        r = requests.get(httpbin('get'), headers=headers_ok)
        assert r.request.headers['foo'] == headers_ok['foo']

    def test_header_value_not_str(self, httpbin):
        """Ensure the header value is of type string or bytes as
        per discussion in GH issue #3386
        """
        headers_int = {'foo': 3}
        headers_dict = {'bar': {'foo': 'bar'}}
        headers_list = {'baz': ['foo', 'bar']}

        # Test for int
        with pytest.raises(InvalidHeader) as excinfo:
            r = requests.get(httpbin('get'), headers=headers_int)
        assert 'foo' in str(excinfo.value)
        # Test for dict
        with pytest.raises(InvalidHeader) as excinfo:
            r = requests.get(httpbin('get'), headers=headers_dict)
        assert 'bar' in str(excinfo.value)
        # Test for list
        with pytest.raises(InvalidHeader) as excinfo:
            r = requests.get(httpbin('get'), headers=headers_list)
        assert 'baz' in str(excinfo.value)

    def test_header_no_return_chars(self, httpbin):
        """Ensure that a header containing return character sequences raise an
        exception. Otherwise, multiple headers are created from single string.
        """
        headers_ret = {'foo': 'bar\r\nbaz: qux'}
        headers_lf = {'foo': 'bar\nbaz: qux'}
        headers_cr = {'foo': 'bar\rbaz: qux'}

        # Test for newline
        with pytest.raises(InvalidHeader):
            r = requests.get(httpbin('get'), headers=headers_ret)
        # Test for line feed
        with pytest.raises(InvalidHeader):
            r = requests.get(httpbin('get'), headers=headers_lf)
        # Test for carriage return
        with pytest.raises(InvalidHeader):
            r = requests.get(httpbin('get'), headers=headers_cr)

    def test_header_no_leading_space(self, httpbin):
        """Ensure headers containing leading whitespace raise
        InvalidHeader Error before sending.
        """
        headers_space = {'foo': ' bar'}
        headers_tab = {'foo': '   bar'}

        # Test for whitespace
        with pytest.raises(InvalidHeader):
            r = requests.get(httpbin('get'), headers=headers_space)
        # Test for tab
        with pytest.raises(InvalidHeader):
            r = requests.get(httpbin('get'), headers=headers_tab)

    @pytest.mark.parametrize('files', ('foo', b'foo', bytearray(b'foo')))
    def test_can_send_objects_with_files(self, httpbin, files):
        data = {'a': 'this is a string'}
        files = {'b': files}
        r = requests.Request('POST', httpbin('post'), data=data, files=files)
        p = r.prepare()
        assert 'multipart/form-data' in p.headers['Content-Type']

    def test_can_send_file_object_with_non_string_filename(self, httpbin):
        f = io.BytesIO()
        f.name = 2
        r = requests.Request('POST', httpbin('post'), files={'f': f})
        p = r.prepare()

        assert 'multipart/form-data' in p.headers['Content-Type']

    def test_autoset_header_values_are_native(self, httpbin):
        data = 'this is a string'
        length = '16'
        req = requests.Request('POST', httpbin('post'), data=data)
        p = req.prepare()

        assert p.headers['Content-Length'] == length

    def test_nonhttp_schemes_dont_check_URLs(self):
        test_urls = (
            'data:image/gif;base64,R0lGODlhAQABAHAAACH5BAUAAAAALAAAAAABAAEAAAICRAEAOw==',
            'file:///etc/passwd',
            'magnet:?xt=urn:btih:be08f00302bc2d1d3cfa3af02024fa647a271431',
        )
        for test_url in test_urls:
            req = requests.Request('GET', test_url)
            preq = req.prepare()
            assert test_url == preq.url

    @pytest.mark.xfail(raises=ConnectionError)
    def test_auth_is_stripped_on_redirect_off_host(self, httpbin):
        r = requests.get(
            httpbin('redirect-to'),
            params={'url': 'http://www.google.co.uk'},
            auth=('user', 'pass'),
        )
        assert r.history[0].request.headers['Authorization']
        assert not r.request.headers.get('Authorization', '')

    def test_auth_is_retained_for_redirect_on_host(self, httpbin):
        r = requests.get(httpbin('redirect/1'), auth=('user', 'pass'))
        h1 = r.history[0].request.headers['Authorization']
        h2 = r.request.headers['Authorization']

        assert h1 == h2

    def test_manual_redirect_with_partial_body_read(self, httpbin):
        s = requests.Session()
        r1 = s.get(httpbin('redirect/2'), allow_redirects=False, stream=True)
        assert r1.is_redirect
        rg = s.resolve_redirects(r1, r1.request, stream=True)

        # read only the first eight bytes of the response body,
        # then follow the redirect
        r1.iter_content(8)
        r2 = next(rg)
        assert r2.is_redirect

        # read all of the response via iter_content,
        # then follow the redirect
        for _ in r2.iter_content():
            pass
        r3 = next(rg)
        assert not r3.is_redirect

    def test_prepare_body_position_non_stream(self):
        data = b'the data'
        s = requests.Session()
        prep = requests.Request('GET', 'http://example.com', data=data).prepare()
        assert prep._body_position is None

    def test_rewind_body(self):
        data = io.BytesIO(b'the data')
        s = requests.Session()
        prep = requests.Request('GET', 'http://example.com', data=data).prepare()
        assert prep._body_position == 0
        assert prep.body.read() == b'the data'

        # the data has all been read
        assert prep.body.read() == b''

        # rewind it back
        requests.utils.rewind_body(prep)
        assert prep.body.read() == b'the data'

    def test_rewind_partially_read_body(self):
        data = io.BytesIO(b'the data')
        s = requests.Session()
        data.read(4)  # read some data
        prep = requests.Request('GET', 'http://example.com', data=data).prepare()
        assert prep._body_position == 4
        assert prep.body.read() == b'data'

        # the data has all been read
        assert prep.body.read() == b''

        # rewind it back
        requests.utils.rewind_body(prep)
        assert prep.body.read() == b'data'

    def test_rewind_body_no_seek(self):
        class BadFileObj:
            def __init__(self, data):
                self.data = data

            def tell(self):
                return 0

            def __iter__(self):
                return

        data = BadFileObj('the data')
        s = requests.Session()
        prep = requests.Request('GET', 'http://example.com', data=data).prepare()
        assert prep._body_position == 0

        with pytest.raises(UnrewindableBodyError) as e:
            requests.utils.rewind_body(prep)

        assert 'Unable to rewind request body' in str(e)

    def test_rewind_body_failed_seek(self):
        class BadFileObj:
            def __init__(self, data):
                self.data = data

            def tell(self):
                return 0

            def seek(self, pos, whence=0):
                raise OSError()

            def __iter__(self):
                return

        data = BadFileObj('the data')
        s = requests.Session()
        prep = requests.Request('GET', 'http://example.com', data=data).prepare()
        assert prep._body_position == 0

        with pytest.raises(UnrewindableBodyError) as e:
            requests.utils.rewind_body(prep)

        assert 'error occurred when rewinding request body' in str(e)

    def test_rewind_body_failed_tell(self):
        class BadFileObj:
            def __init__(self, data):
                self.data = data

            def tell(self):
                raise OSError()

            def __iter__(self):
                return

        data = BadFileObj('the data')
        s = requests.Session()
        prep = requests.Request('GET', 'http://example.com', data=data).prepare()
        assert prep._body_position is not None

        with pytest.raises(UnrewindableBodyError) as e:
            requests.utils.rewind_body(prep)

        assert 'Unable to rewind request body' in str(e)

    def _patch_adapter_gzipped_redirect(self, session, url):
        adapter = session.get_adapter(url=url)
        org_build_response = adapter.build_response
        self._patched_response = False

        def build_response(*args, **kwargs):
            resp = org_build_response(*args, **kwargs)
            if not self._patched_response:
                resp.raw.headers['content-encoding'] = 'gzip'
                self._patched_response = True
            return resp

        adapter.build_response = build_response

    def test_redirect_with_wrong_gzipped_header(self, httpbin):
        s = requests.Session()
        url = httpbin('redirect/1')
        self._patch_adapter_gzipped_redirect(s, url)
        s.get(url)

    @pytest.mark.parametrize(
        'username, password, auth_str', (
            ('test', 'test', 'Basic dGVzdDp0ZXN0'),
            (u'имя'.encode('utf-8'), u'пароль'.encode('utf-8'), 'Basic 0LjQvNGPOtC/0LDRgNC+0LvRjA=='),
        ))
    def test_basic_auth_str_is_always_native(self, username, password, auth_str):
        s = _basic_auth_str(username, password)
        assert isinstance(s, builtin_str)
        assert s == auth_str

    def test_requests_history_is_saved(self, httpbin):
        r = requests.get(httpbin('redirect/5'))
        total = r.history[-1].history
        i = 0
        for item in r.history:
            assert item.history == total[0:i]
            i += 1

    def test_json_param_post_content_type_works(self, httpbin):
        r = requests.post(
            httpbin('post'),
            json={'life': 42}
        )
        assert r.status_code == 200
        assert 'application/json' in r.request.headers['Content-Type']
        assert {'life': 42} == r.json()['json']

    def test_json_param_post_should_not_override_data_param(self, httpbin):
        r = requests.Request(method='POST', url=httpbin('post'),
                             data={'stuff': 'elixr'},
                             json={'music': 'flute'})
        prep = r.prepare()
        assert 'stuff=elixr' == prep.body

    def test_response_iter_lines(self, httpbin):
        r = requests.get(httpbin('stream/4'), stream=True)
        assert r.status_code == 200

        it = r.iter_lines()
        next(it)
        assert len(list(it)) == 3

    def test_response_context_manager(self, httpbin):
        with requests.get(httpbin('stream/4'), stream=True) as response:
            assert isinstance(response, requests.Response)

        assert response.raw.closed

    def test_unconsumed_session_response_closes_connection(self, httpbin):
        s = requests.session()

        with contextlib.closing(s.get(httpbin('stream/4'), stream=True)) as response:
            pass

        assert response._content_consumed is False
        assert response.raw.closed

    @pytest.mark.xfail
    def test_response_iter_lines_reentrant(self, httpbin):
        """Response.iter_lines() is not reentrant safe"""
        r = requests.get(httpbin('stream/4'), stream=True)
        assert r.status_code == 200

        next(r.iter_lines())
        assert len(list(r.iter_lines())) == 3

    def test_session_close_proxy_clear(self, mocker):
        proxies = {
          'one': mocker.Mock(),
          'two': mocker.Mock(),
        }
        session = requests.Session()
        mocker.patch.dict(session.adapters['http://'].proxy_manager, proxies)
        session.close()
        proxies['one'].clear.assert_called_once_with()
        proxies['two'].clear.assert_called_once_with()

    def test_proxy_auth(self, httpbin):
        adapter = HTTPAdapter()
        headers = adapter.proxy_headers("http://user:pass@httpbin.org")
        assert headers == {'Proxy-Authorization': 'Basic dXNlcjpwYXNz'}

    def test_proxy_auth_empty_pass(self, httpbin):
        adapter = HTTPAdapter()
        headers = adapter.proxy_headers("http://user:@httpbin.org")
        assert headers == {'Proxy-Authorization': 'Basic dXNlcjo='}

    def test_response_json_when_content_is_None(self, httpbin):
        r = requests.get(httpbin('/status/204'))
        # Make sure r.content is None
        r.status_code = 0
        r._content = False
        r._content_consumed = False

        assert r.content is None
        with pytest.raises(ValueError):
            r.json()

    def test_response_without_release_conn(self):
        """Test `close` call for non-urllib3-like raw objects.
        Should work when `release_conn` attr doesn't exist on `response.raw`.
        """
        resp = requests.Response()
        resp.raw = StringIO.StringIO('test')
        assert not resp.raw.closed
        resp.close()
        assert resp.raw.closed

    def test_empty_stream_with_auth_does_not_set_content_length_header(self, httpbin):
        """Ensure that a byte stream with size 0 will not set both a Content-Length
        and Transfer-Encoding header.
        """
        auth = ('user', 'pass')
        url = httpbin('post')
        file_obj = io.BytesIO(b'')
        r = requests.Request('POST', url, auth=auth, data=file_obj)
        prepared_request = r.prepare()
        assert 'Transfer-Encoding' in prepared_request.headers
        assert 'Content-Length' not in prepared_request.headers

    def test_stream_with_auth_does_not_set_transfer_encoding_header(self, httpbin):
        """Ensure that a byte stream with size > 0 will not set both a Content-Length
        and Transfer-Encoding header.
        """
        auth = ('user', 'pass')
        url = httpbin('post')
        file_obj = io.BytesIO(b'test data')
        r = requests.Request('POST', url, auth=auth, data=file_obj)
        prepared_request = r.prepare()
        assert 'Transfer-Encoding' not in prepared_request.headers
        assert 'Content-Length' in prepared_request.headers

    def test_chunked_upload_does_not_set_content_length_header(self, httpbin):
        """Ensure that requests with a generator body stream using
        Transfer-Encoding: chunked, not a Content-Length header.
        """
        data = (i for i in [b'a', b'b', b'c'])
        url = httpbin('post')
        r = requests.Request('POST', url, data=data)
        prepared_request = r.prepare()
        assert 'Transfer-Encoding' in prepared_request.headers
        assert 'Content-Length' not in prepared_request.headers

    def test_custom_redirect_mixin(self, httpbin):
        """Tests a custom mixin to overwrite ``get_redirect_target``.

        Ensures a subclassed ``requests.Session`` can handle a certain type of
        malformed redirect responses.

        1. original request receives a proper response: 302 redirect
        2. following the redirect, a malformed response is given:
            status code = HTTP 200
            location = alternate url
        3. the custom session catches the edge case and follows the redirect
        """
        url_final = httpbin('html')
        querystring_malformed = urlencode({'location': url_final})
        url_redirect_malformed = httpbin('response-headers?%s' % querystring_malformed)
        querystring_redirect = urlencode({'url': url_redirect_malformed})
        url_redirect = httpbin('redirect-to?%s' % querystring_redirect)
        urls_test = [url_redirect,
                     url_redirect_malformed,
                     url_final,
                     ]

        class CustomRedirectSession(requests.Session):
            def get_redirect_target(self, resp):
                # default behavior
                if resp.is_redirect:
                    return resp.headers['location']
                # edge case - check to see if 'location' is in headers anyways
                location = resp.headers.get('location')
                if location and (location != resp.url):
                    return location
                return None

        session = CustomRedirectSession()
        r = session.get(urls_test[0])
        assert len(r.history) == 2
        assert r.status_code == 200
        assert r.history[0].status_code == 302
        assert r.history[0].is_redirect
        assert r.history[1].status_code == 200
        assert not r.history[1].is_redirect
        assert r.url == urls_test[2]


class TestCaseInsensitiveDict:

    @pytest.fixture(autouse=True)
    def setup(self):
        """CaseInsensitiveDict instance with "Accept" header."""
        self.case_insensitive_dict = CaseInsensitiveDict()
        self.case_insensitive_dict['Accept'] = 'application/json'

    def test_list(self):
        assert list(self.case_insensitive_dict) == ['Accept']

    possible_keys = pytest.mark.parametrize('key', ('accept', 'ACCEPT', 'aCcEpT', 'Accept'))

    @possible_keys
    def test_getitem(self, key):
        assert self.case_insensitive_dict[key] == 'application/json'

    @possible_keys
    def test_delitem(self, key):
        del self.case_insensitive_dict[key]
        assert key not in self.case_insensitive_dict

    def test_lower_items(self):
        assert list(self.case_insensitive_dict.lower_items()) == [('accept', 'application/json')]

    def test_repr(self):
        assert repr(self.case_insensitive_dict) == "{'Accept': 'application/json'}"

    def test_copy(self):
        copy = self.case_insensitive_dict.copy()
        assert copy is not self.case_insensitive_dict
        assert copy == self.case_insensitive_dict

    @pytest.mark.parametrize(
        'other, result', (
            ({'AccePT': 'application/json'}, True),
            ({}, False),
            (None, False)
        )
    )
    def test_instance_equality(self, other, result):
        assert (self.case_insensitive_dict == other) is result


class TestPreparingURLs(object):
    @pytest.mark.parametrize(
        'url,expected',
        (
            ('http://google.com', 'http://google.com/'),
            (u'http://ジェーピーニック.jp', u'http://xn--hckqz9bzb1cyrb.jp/'),
            (u'http://xn--n3h.net/', u'http://xn--n3h.net/'),
            (
                u'http://ジェーピーニック.jp'.encode('utf-8'),
                u'http://xn--hckqz9bzb1cyrb.jp/'
            ),
            (
                u'http://straße.de/straße',
                u'http://xn--strae-oqa.de/stra%C3%9Fe'
            ),
            (
                u'http://straße.de/straße'.encode('utf-8'),
                u'http://xn--strae-oqa.de/stra%C3%9Fe'
            ),
            (
                u'http://Königsgäßchen.de/straße',
                u'http://xn--knigsgchen-b4a3dun.de/stra%C3%9Fe'
            ),
            (
                u'http://Königsgäßchen.de/straße'.encode('utf-8'),
                u'http://xn--knigsgchen-b4a3dun.de/stra%C3%9Fe'
            ),
            (
                b'http://xn--n3h.net/',
                u'http://xn--n3h.net/'
            ),
            (
                b'http://[1200:0000:ab00:1234:0000:2552:7777:1313]:12345/',
                u'http://[1200:0000:ab00:1234:0000:2552:7777:1313]:12345/'
            ),
            (
                u'http://[1200:0000:ab00:1234:0000:2552:7777:1313]:12345/',
                u'http://[1200:0000:ab00:1234:0000:2552:7777:1313]:12345/'
            )
        )
    )
    def test_preparing_url(self, url, expected):
        r = requests.Request('GET', url=url)
        p = r.prepare()
        assert p.url == expected

    @pytest.mark.parametrize(
        'url',
        (
            b"http://*.google.com",
            b"http://*",
            u"http://*.google.com",
            u"http://*",
            u"http://☃.net/"
        )
    )
    def test_preparing_bad_url(self, url):
        r = requests.Request('GET', url=url)
        with pytest.raises(requests.exceptions.InvalidURL):
            r.prepare()

    @pytest.mark.parametrize(
        'input, expected',
        (
            (
                b"http+unix://%2Fvar%2Frun%2Fsocket/path%7E",
                u"http+unix://%2Fvar%2Frun%2Fsocket/path~",
            ),
            (
                u"http+unix://%2Fvar%2Frun%2Fsocket/path%7E",
                u"http+unix://%2Fvar%2Frun%2Fsocket/path~",
            ),
            (
                b"mailto:user@example.org",
                u"mailto:user@example.org",
            ),
            (
                u"mailto:user@example.org",
                u"mailto:user@example.org",
            ),
            (
                b"data:SSDimaUgUHl0aG9uIQ==",
                u"data:SSDimaUgUHl0aG9uIQ==",
            )
        )
    )
    def test_url_mutation(self, input, expected):
        """
        This test validates that we correctly exclude some URLs from
        preparation, and that we handle others. Specifically, it tests that
        any URL whose scheme doesn't begin with "http" is left alone, and
        those whose scheme *does* begin with "http" are mutated.
        """
        r = requests.Request('GET', url=input)
        p = r.prepare()
        assert p.url == expected

    @pytest.mark.parametrize(
        'input, params, expected',
        (
            (
                b"http+unix://%2Fvar%2Frun%2Fsocket/path",
                {"key": "value"},
                u"http+unix://%2Fvar%2Frun%2Fsocket/path?key=value",
            ),
            (
                u"http+unix://%2Fvar%2Frun%2Fsocket/path",
                {"key": "value"},
                u"http+unix://%2Fvar%2Frun%2Fsocket/path?key=value",
            ),
            (
                b"mailto:user@example.org",
                {"key": "value"},
                u"mailto:user@example.org",
            ),
            (
                u"mailto:user@example.org",
                {"key": "value"},
                u"mailto:user@example.org",
            ),
        )
    )
    def test_parameters_for_nonstandard_schemes(self, input, params, expected):
        """
        Setting parameters for nonstandard schemes is allowed if those schemes
        begin with "http", and is forbidden otherwise.
        """
        r = requests.Request('GET', url=input, params=params)
        p = r.prepare()
        assert p.url == expected

class TestCaseInsensitiveDict:

    @pytest.fixture(autouse=True)
    def setup(self):
        """CaseInsensitiveDict instance with "Accept" header."""
        self.case_insensitive_dict = CaseInsensitiveDict()
        self.case_insensitive_dict['Accept'] = 'application/json'

    def test_list(self):
        assert list(self.case_insensitive_dict) == ['Accept']

    possible_keys = pytest.mark.parametrize('key', ('accept', 'ACCEPT', 'aCcEpT', 'Accept'))

    @possible_keys
    def test_getitem(self, key):
        assert self.case_insensitive_dict[key] == 'application/json'

    @possible_keys
    def test_delitem(self, key):
        del self.case_insensitive_dict[key]
        assert key not in self.case_insensitive_dict

    def test_lower_items(self):
        assert list(self.case_insensitive_dict.lower_items()) == [('accept', 'application/json')]

    def test_repr(self):
        assert repr(self.case_insensitive_dict) == "{'Accept': 'application/json'}"

    def test_copy(self):
        copy = self.case_insensitive_dict.copy()
        assert copy is not self.case_insensitive_dict
        assert copy == self.case_insensitive_dict

    @pytest.mark.parametrize(
        'other, result', (
            ({'AccePT': 'application/json'}, True),
            ({}, False),
            (None, False)
        )
    )
    def test_instance_equality(self, other, result):
        assert (self.case_insensitive_dict == other) is result


class TestCaseInsensitiveDict:

    @pytest.mark.parametrize(
        'cid', (
            CaseInsensitiveDict({'Foo': 'foo', 'BAr': 'bar'}),
            CaseInsensitiveDict([('Foo', 'foo'), ('BAr', 'bar')]),
            CaseInsensitiveDict(FOO='foo', BAr='bar'),
        ))
    def test_init(self, cid):
        assert len(cid) == 2
        assert 'foo' in cid
        assert 'bar' in cid

    def test_docstring_example(self):
        cid = CaseInsensitiveDict()
        cid['Accept'] = 'application/json'
        assert cid['aCCEPT'] == 'application/json'
        assert list(cid) == ['Accept']

    def test_len(self):
        cid = CaseInsensitiveDict({'a': 'a', 'b': 'b'})
        cid['A'] = 'a'
        assert len(cid) == 2

    def test_getitem(self):
        cid = CaseInsensitiveDict({'Spam': 'blueval'})
        assert cid['spam'] == 'blueval'
        assert cid['SPAM'] == 'blueval'

    def test_fixes_649(self):
        """__setitem__ should behave case-insensitively."""
        cid = CaseInsensitiveDict()
        cid['spam'] = 'oneval'
        cid['Spam'] = 'twoval'
        cid['sPAM'] = 'redval'
        cid['SPAM'] = 'blueval'
        assert cid['spam'] == 'blueval'
        assert cid['SPAM'] == 'blueval'
        assert list(cid.keys()) == ['SPAM']

    def test_delitem(self):
        cid = CaseInsensitiveDict()
        cid['Spam'] = 'someval'
        del cid['sPam']
        assert 'spam' not in cid
        assert len(cid) == 0

    def test_contains(self):
        cid = CaseInsensitiveDict()
        cid['Spam'] = 'someval'
        assert 'Spam' in cid
        assert 'spam' in cid
        assert 'SPAM' in cid
        assert 'sPam' in cid
        assert 'notspam' not in cid

    def test_get(self):
        cid = CaseInsensitiveDict()
        cid['spam'] = 'oneval'
        cid['SPAM'] = 'blueval'
        assert cid.get('spam') == 'blueval'
        assert cid.get('SPAM') == 'blueval'
        assert cid.get('sPam') == 'blueval'
        assert cid.get('notspam', 'default') == 'default'

    def test_update(self):
        cid = CaseInsensitiveDict()
        cid['spam'] = 'blueval'
        cid.update({'sPam': 'notblueval'})
        assert cid['spam'] == 'notblueval'
        cid = CaseInsensitiveDict({'Foo': 'foo', 'BAr': 'bar'})
        cid.update({'fOO': 'anotherfoo', 'bAR': 'anotherbar'})
        assert len(cid) == 2
        assert cid['foo'] == 'anotherfoo'
        assert cid['bar'] == 'anotherbar'

    def test_update_retains_unchanged(self):
        cid = CaseInsensitiveDict({'foo': 'foo', 'bar': 'bar'})
        cid.update({'foo': 'newfoo'})
        assert cid['bar'] == 'bar'

    def test_iter(self):
        cid = CaseInsensitiveDict({'Spam': 'spam', 'Eggs': 'eggs'})
        keys = frozenset(['Spam', 'Eggs'])
        assert frozenset(iter(cid)) == keys

    def test_equality(self):
        cid = CaseInsensitiveDict({'SPAM': 'blueval', 'Eggs': 'redval'})
        othercid = CaseInsensitiveDict({'spam': 'blueval', 'eggs': 'redval'})
        assert cid == othercid
        del othercid['spam']
        assert cid != othercid
        assert cid == {'spam': 'blueval', 'eggs': 'redval'}
        assert cid != object()

    def test_setdefault(self):
        cid = CaseInsensitiveDict({'Spam': 'blueval'})
        assert cid.setdefault('spam', 'notblueval') == 'blueval'
        assert cid.setdefault('notspam', 'notblueval') == 'notblueval'

    def test_lower_items(self):
        cid = CaseInsensitiveDict({
            'Accept': 'application/json',
            'user-Agent': 'requests',
        })
        keyset = frozenset(lowerkey for lowerkey, v in cid.lower_items())
        lowerkeyset = frozenset(['accept', 'user-agent'])
        assert keyset == lowerkeyset

    def test_preserve_key_case(self):
        cid = CaseInsensitiveDict({
            'Accept': 'application/json',
            'user-Agent': 'requests',
        })
        keyset = frozenset(['Accept', 'user-Agent'])
        assert frozenset(i[0] for i in cid.items()) == keyset
        assert frozenset(cid.keys()) == keyset
        assert frozenset(cid) == keyset

    def test_preserve_last_key_case(self):
        cid = CaseInsensitiveDict({
            'Accept': 'application/json',
            'user-Agent': 'requests',
        })
        cid.update({'ACCEPT': 'application/json'})
        cid['USER-AGENT'] = 'requests'
        keyset = frozenset(['ACCEPT', 'USER-AGENT'])
        assert frozenset(i[0] for i in cid.items()) == keyset
        assert frozenset(cid.keys()) == keyset
        assert frozenset(cid) == keyset

    def test_copy(self):
        cid = CaseInsensitiveDict({
            'Accept': 'application/json',
            'user-Agent': 'requests',
        })
        cid_copy = cid.copy()
        assert cid == cid_copy
        cid['changed'] = True
        assert cid != cid_copy


@pytest.mark.skipif(sys.version_info < (2,7), reason="Only run on Python 2.7+")
def test_system_ssl():
    """Verify we're actually setting system_ssl when it should be available."""
    assert info()['system_ssl']['version'] != ''


def test_can_access_urllib3_attribute():
    requests.packages.urllib3


class TestAddressInNetwork:

    def test_valid(self):
        assert address_in_network('192.168.1.1', '192.168.1.0/24')

    def test_invalid(self):
        assert not address_in_network('172.16.0.1', '192.168.1.0/24')


def hook(value):
    return value[1:]


@pytest.mark.parametrize("var,scheme", _proxy_combos)
def test_use_proxy_from_environment(httpbin, var, scheme):
    url = "{0}://httpbin.org".format(scheme)
    fake_proxy = Server()  # do nothing with the requests; just close the socket
    with fake_proxy as (host, port):
        proxy_url = "socks5://{0}:{1}".format(host, port)
        kwargs = {var: proxy_url}
        with override_environ(**kwargs):
            # fake proxy's lack of response will cause a ConnectionError
            with pytest.raises(requests.exceptions.ConnectionError):
                requests.get(url)

        # the fake proxy received a request
        assert len(fake_proxy.handler_results) == 1

        # it had actual content (not checking for SOCKS protocol for now)
        assert len(fake_proxy.handler_results[0]) > 0


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

_proxy_combos = []
for prefix, schemes in _schemes_by_var_prefix:
    for scheme in schemes:
        _proxy_combos.append(("{0}_proxy".format(prefix), scheme))

_proxy_combos += [(var.upper(), scheme) for var, scheme in _proxy_combos]


@pytest.mark.skipif(sys.version_info[:2] != (2,6), reason="Only run on Python 2.6")
def test_system_ssl_py26():
    """OPENSSL_VERSION_NUMBER isn't provided in Python 2.6, verify we don't
    blow up in this case.
    """
    assert info()['system_ssl'] == {'version': ''}


class TestMorselToCookieMaxAge:

    """Tests for morsel_to_cookie when morsel contains max-age."""

    def test_max_age_valid_int(self):
        """Test case where a valid max age in seconds is passed."""

        morsel = Morsel()
        morsel['max-age'] = 60
        cookie = morsel_to_cookie(morsel)
        assert isinstance(cookie.expires, int)

    def test_max_age_invalid_str(self):
        """Test case where a invalid max age is passed."""

        morsel = Morsel()
        morsel['max-age'] = 'woops'
        with pytest.raises(TypeError):
            morsel_to_cookie(morsel)


def test_can_access_urllib3_attribute():
    requests.packages.urllib3


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


class TestIsIPv4Address:

    def test_valid(self):
        assert is_ipv4_address('8.8.8.8')

    @pytest.mark.parametrize('value', ('8.8.8.8.8', 'localhost.localdomain'))
    def test_invalid(self, value):
        assert not is_ipv4_address(value)


def test_can_access_idna_attribute():
    requests.packages.idna


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


def hook(value):
    return value[1:]


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

class TestPreparingURLs(object):
    @pytest.mark.parametrize(
        'url,expected',
        (
            ('http://google.com', 'http://google.com/'),
            (u'http://ジェーピーニック.jp', u'http://xn--hckqz9bzb1cyrb.jp/'),
            (u'http://xn--n3h.net/', u'http://xn--n3h.net/'),
            (
                u'http://ジェーピーニック.jp'.encode('utf-8'),
                u'http://xn--hckqz9bzb1cyrb.jp/'
            ),
            (
                u'http://straße.de/straße',
                u'http://xn--strae-oqa.de/stra%C3%9Fe'
            ),
            (
                u'http://straße.de/straße'.encode('utf-8'),
                u'http://xn--strae-oqa.de/stra%C3%9Fe'
            ),
            (
                u'http://Königsgäßchen.de/straße',
                u'http://xn--knigsgchen-b4a3dun.de/stra%C3%9Fe'
            ),
            (
                u'http://Königsgäßchen.de/straße'.encode('utf-8'),
                u'http://xn--knigsgchen-b4a3dun.de/stra%C3%9Fe'
            ),
            (
                b'http://xn--n3h.net/',
                u'http://xn--n3h.net/'
            ),
            (
                b'http://[1200:0000:ab00:1234:0000:2552:7777:1313]:12345/',
                u'http://[1200:0000:ab00:1234:0000:2552:7777:1313]:12345/'
            ),
            (
                u'http://[1200:0000:ab00:1234:0000:2552:7777:1313]:12345/',
                u'http://[1200:0000:ab00:1234:0000:2552:7777:1313]:12345/'
            )
        )
    )
    def test_preparing_url(self, url, expected):
        r = requests.Request('GET', url=url)
        p = r.prepare()
        assert p.url == expected

    @pytest.mark.parametrize(
        'url',
        (
            b"http://*.google.com",
            b"http://*",
            u"http://*.google.com",
            u"http://*",
            u"http://☃.net/"
        )
    )
    def test_preparing_bad_url(self, url):
        r = requests.Request('GET', url=url)
        with pytest.raises(requests.exceptions.InvalidURL):
            r.prepare()

    @pytest.mark.parametrize(
        'input, expected',
        (
            (
                b"http+unix://%2Fvar%2Frun%2Fsocket/path%7E",
                u"http+unix://%2Fvar%2Frun%2Fsocket/path~",
            ),
            (
                u"http+unix://%2Fvar%2Frun%2Fsocket/path%7E",
                u"http+unix://%2Fvar%2Frun%2Fsocket/path~",
            ),
            (
                b"mailto:user@example.org",
                u"mailto:user@example.org",
            ),
            (
                u"mailto:user@example.org",
                u"mailto:user@example.org",
            ),
            (
                b"data:SSDimaUgUHl0aG9uIQ==",
                u"data:SSDimaUgUHl0aG9uIQ==",
            )
        )
    )
    def test_url_mutation(self, input, expected):
        """
        This test validates that we correctly exclude some URLs from
        preparation, and that we handle others. Specifically, it tests that
        any URL whose scheme doesn't begin with "http" is left alone, and
        those whose scheme *does* begin with "http" are mutated.
        """
        r = requests.Request('GET', url=input)
        p = r.prepare()
        assert p.url == expected

    @pytest.mark.parametrize(
        'input, params, expected',
        (
            (
                b"http+unix://%2Fvar%2Frun%2Fsocket/path",
                {"key": "value"},
                u"http+unix://%2Fvar%2Frun%2Fsocket/path?key=value",
            ),
            (
                u"http+unix://%2Fvar%2Frun%2Fsocket/path",
                {"key": "value"},
                u"http+unix://%2Fvar%2Frun%2Fsocket/path?key=value",
            ),
            (
                b"mailto:user@example.org",
                {"key": "value"},
                u"mailto:user@example.org",
            ),
            (
                u"mailto:user@example.org",
                {"key": "value"},
                u"mailto:user@example.org",
            ),
        )
    )
    def test_parameters_for_nonstandard_schemes(self, input, params, expected):
        """
        Setting parameters for nonstandard schemes is allowed if those schemes
        begin with "http", and is forbidden otherwise.
        """
        r = requests.Request('GET', url=input, params=params)
        p = r.prepare()
        assert p.url == expected

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


class TestPreparingURLs(object):
    @pytest.mark.parametrize(
        'url,expected',
        (
            ('http://google.com', 'http://google.com/'),
            (u'http://ジェーピーニック.jp', u'http://xn--hckqz9bzb1cyrb.jp/'),
            (u'http://xn--n3h.net/', u'http://xn--n3h.net/'),
            (
                u'http://ジェーピーニック.jp'.encode('utf-8'),
                u'http://xn--hckqz9bzb1cyrb.jp/'
            ),
            (
                u'http://straße.de/straße',
                u'http://xn--strae-oqa.de/stra%C3%9Fe'
            ),
            (
                u'http://straße.de/straße'.encode('utf-8'),
                u'http://xn--strae-oqa.de/stra%C3%9Fe'
            ),
            (
                u'http://Königsgäßchen.de/straße',
                u'http://xn--knigsgchen-b4a3dun.de/stra%C3%9Fe'
            ),
            (
                u'http://Königsgäßchen.de/straße'.encode('utf-8'),
                u'http://xn--knigsgchen-b4a3dun.de/stra%C3%9Fe'
            ),
            (
                b'http://xn--n3h.net/',
                u'http://xn--n3h.net/'
            ),
            (
                b'http://[1200:0000:ab00:1234:0000:2552:7777:1313]:12345/',
                u'http://[1200:0000:ab00:1234:0000:2552:7777:1313]:12345/'
            ),
            (
                u'http://[1200:0000:ab00:1234:0000:2552:7777:1313]:12345/',
                u'http://[1200:0000:ab00:1234:0000:2552:7777:1313]:12345/'
            )
        )
    )
    def test_preparing_url(self, url, expected):
        r = requests.Request('GET', url=url)
        p = r.prepare()
        assert p.url == expected

    @pytest.mark.parametrize(
        'url',
        (
            b"http://*.google.com",
            b"http://*",
            u"http://*.google.com",
            u"http://*",
            u"http://☃.net/"
        )
    )
    def test_preparing_bad_url(self, url):
        r = requests.Request('GET', url=url)
        with pytest.raises(requests.exceptions.InvalidURL):
            r.prepare()

    @pytest.mark.parametrize(
        'input, expected',
        (
            (
                b"http+unix://%2Fvar%2Frun%2Fsocket/path%7E",
                u"http+unix://%2Fvar%2Frun%2Fsocket/path~",
            ),
            (
                u"http+unix://%2Fvar%2Frun%2Fsocket/path%7E",
                u"http+unix://%2Fvar%2Frun%2Fsocket/path~",
            ),
            (
                b"mailto:user@example.org",
                u"mailto:user@example.org",
            ),
            (
                u"mailto:user@example.org",
                u"mailto:user@example.org",
            ),
            (
                b"data:SSDimaUgUHl0aG9uIQ==",
                u"data:SSDimaUgUHl0aG9uIQ==",
            )
        )
    )
    def test_url_mutation(self, input, expected):
        """
        This test validates that we correctly exclude some URLs from
        preparation, and that we handle others. Specifically, it tests that
        any URL whose scheme doesn't begin with "http" is left alone, and
        those whose scheme *does* begin with "http" are mutated.
        """
        r = requests.Request('GET', url=input)
        p = r.prepare()
        assert p.url == expected

    @pytest.mark.parametrize(
        'input, params, expected',
        (
            (
                b"http+unix://%2Fvar%2Frun%2Fsocket/path",
                {"key": "value"},
                u"http+unix://%2Fvar%2Frun%2Fsocket/path?key=value",
            ),
            (
                u"http+unix://%2Fvar%2Frun%2Fsocket/path",
                {"key": "value"},
                u"http+unix://%2Fvar%2Frun%2Fsocket/path?key=value",
            ),
            (
                b"mailto:user@example.org",
                {"key": "value"},
                u"mailto:user@example.org",
            ),
            (
                u"mailto:user@example.org",
                {"key": "value"},
                u"mailto:user@example.org",
            ),
        )
    )
    def test_parameters_for_nonstandard_schemes(self, input, params, expected):
        """
        Setting parameters for nonstandard schemes is allowed if those schemes
        begin with "http", and is forbidden otherwise.
        """
        r = requests.Request('GET', url=input, params=params)
        p = r.prepare()
        assert p.url == expected
