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


def test_default_hooks():
    assert hooks.default_hooks() == {'response': []}

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

    # def test_binary_put(self):
    #     request = requests.Request('PUT', 'http://example.com',
    #                                data=u"ööö".encode("utf-8")).prepare()
    #     assert isinstance(request.body, bytes)

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

    # @pytest.mark.parametrize(
    #     'username, password', (
    #         ('user', 'pass'),
    #         (u'имя'.encode('utf-8'), u'пароль'.encode('utf-8')),
    #         (42, 42),
    #         (None, None),
    #     ))
    # def test_set_basicauth(self, httpbin, username, password):
    #     auth = (username, password)
    #     url = httpbin('get')
    #
    #     r = requests.Request('GET', url, auth=auth)
    #     p = r.prepare()
    #
    #     assert p.headers['Authorization'] == _basic_auth_str(username, password)

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

    # def test_conflicting_post_params(self, httpbin):
    #     url = httpbin('post')
    #     with open('Pipfile') as f:
    #         pytest.raises(ValueError, "requests.post(url, data='[{\"some\": \"data\"}]', files={'some': f})")
    #         pytest.raises(ValueError, "requests.post(url, data=u('[{\"some\": \"data\"}]'), files={'some': f})")

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

    # @pytest.mark.parametrize(
    #     'data', (
    #         {'stuff': u('ëlïxr')},
    #         {'stuff': u('ëlïxr').encode('utf-8')},
    #         {'stuff': 'elixr'},
    #         {'stuff': 'elixr'.encode('utf-8')},
    #     ))
    # def test_unicode_multipart_post(self, httpbin, data):
    #     r = requests.post(httpbin('post'),
    #         data=data,
    #         files={'file': ('test_requests.py', open(__file__, 'rb'))})
    #     assert r.status_code == 200

    def test_unicode_multipart_post_fieldnames(self, httpbin):
        filename = os.path.splitext(__file__)[0] + '.py'
        r = requests.Request(
            method='POST', url=httpbin('post'),
            data={'stuff'.encode('utf-8'): 'elixr'},
            files={'file': ('test_requests.py', open(filename, 'rb'))})
        prep = r.prepare()
        assert b'name="stuff"' in prep.body
        assert b'name="b\'stuff\'"' not in prep.body

    # def test_unicode_method_name(self, httpbin):
    #     files = {'file': open(__file__, 'rb')}
    #     r = requests.request(
    #         method=u('POST'), url=httpbin('post'), files=files)
    #     assert r.status_code == 200

    # def test_unicode_method_name_with_request_object(self, httpbin):
    #     files = {'file': open(__file__, 'rb')}
    #     s = requests.Session()
    #     req = requests.Request(u('POST'), httpbin('post'), files=files)
    #     prep = s.prepare_request(req)
    #     assert isinstance(prep.method, builtin_str)
    #     assert prep.method == 'POST'
    #
    #     resp = s.send(prep)
    #     assert resp.status_code == 200

    # def test_non_prepared_request_error(self):
    #     s = requests.Session()
    #     req = requests.Request(u('POST'), '/')
    #
    #     with pytest.raises(ValueError) as e:
    #         s.send(req)
    #     assert str(e.value) == 'You can only send PreparedRequests.'

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

    # def test_response_reason_unicode(self):
    #     # check for unicode HTTP status
    #     r = requests.Response()
    #     r.url = u'unicode URL'
    #     r.reason = u'Komponenttia ei löydy'.encode('utf-8')
    #     r.status_code = 404
    #     r.encoding = None
    #     assert not r.ok  # old behaviour - crashes here

    # def test_response_reason_unicode_fallback(self):
    #     # check raise_status falls back to ISO-8859-1
    #     r = requests.Response()
    #     r.url = 'some url'
    #     reason = u'Komponenttia ei löydy'
    #     r.reason = reason.encode('latin-1')
    #     r.status_code = 500
    #     r.encoding = None
    #     with pytest.raises(requests.exceptions.HTTPError) as e:
    #         r.raise_for_status()
    #     assert reason in e.value.args[0]

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
