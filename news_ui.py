"""Public news feed and password-protected publishing controls."""
import hashlib
import hmac
import os
import time
from datetime import date
import streamlit as st
from news_store import list_posts, save_post, delete_post, StorageError


def setting(name, default=''):
    value = os.environ.get(name, '').strip()
    if value:
        return value
    try:
        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return default


def authorized(expected):
    token = hashlib.sha256(expected.encode()).hexdigest() if expected else ''
    return bool(token and st.session_state.get('news_admin_token') == token
                and time.time() < st.session_state.get('news_admin_until', 0))


@st.cache_data(ttl=60, show_spinner=False)
def cached_posts(repo, branch, token):
    return list_posts(dict(repo=repo, branch=branch, token=token))


def daily_news_tab():
    st.subheader('📰 Daily Financial News')
    st.caption('Daily updates and commentary by Durgesh Shukla')
    path = dict(repo=setting('NEWS_GITHUB_REPO'), branch=setting('NEWS_GITHUB_BRANCH', 'main'), token=setting('NEWS_GITHUB_TOKEN'))
    if not path['repo'] or not path['token']:
        st.info('News storage is not configured yet. Add NEWS_GITHUB_REPO and NEWS_GITHUB_TOKEN in Streamlit Secrets.')
        return
    try:
        posts = cached_posts(path['repo'], path['branch'], path['token'])
    except (StorageError, ValueError) as exc:
        st.error(str(exc))
        return
    if not posts:
        st.info('The first news update is coming soon.')
    page_count = max(1, (len(posts) + 9) // 10)
    page = st.selectbox('News page', range(1, page_count + 1)) if page_count > 1 else 1
    for post in posts[(page-1)*10:page*10]:
        with st.container(border=True):
            st.subheader(post['headline'])
            st.caption(post['published_date'])
            st.markdown(post['body'])
            if post['analysis']:
                st.markdown('**My analysis — why this matters**')
                st.markdown(post['analysis'])
            if post['source_url']:
                st.link_button('Read original source', post['source_url'])
    with st.expander('Admin — manage news'):
        expected = setting('NEWS_ADMIN_PASSWORD')
        if not expected:
            st.caption('Publishing is disabled until NEWS_ADMIN_PASSWORD is configured in private settings.')
            return
        if not authorized(expected):
            with st.form('news_login'):
                password = st.text_input('Admin password', type='password')
                login = st.form_submit_button('Log in')
            if login:
                if time.time() < st.session_state.get('news_retry_after', 0):
                    st.error('Please wait 30 seconds before trying again.')
                elif hmac.compare_digest(password.encode(), expected.encode()):
                    st.session_state.news_admin_token = hashlib.sha256(expected.encode()).hexdigest()
                    st.session_state.news_admin_until = time.time() + 3600
                    st.rerun()
                else:
                    st.session_state.news_retry_after = time.time() + 30
                    st.error('Incorrect password.')
            return
        if st.button('Log out', key='news_logout'):
            st.session_state.pop('news_admin_token', None)
            st.rerun()
        st.caption('Admin session expires after one hour. All submitted posts are public immediately.')
        choices = {p['id']: p for p in posts}
        chosen = st.selectbox('Create or edit', [None] + list(choices),
                              format_func=lambda x: 'New article' if x is None else f"#{x}: {choices[x]['headline']}")
        current = choices.get(chosen, {})
        with st.form(f'news_editor_{chosen}'):
            headline = st.text_input('Headline', value=current.get('headline', ''), max_chars=250)
            published = st.date_input('Publication date', value=date.fromisoformat(current['published_date']) if current else date.today(), max_value=date.today())
            body = st.text_area('News summary / article (Markdown supported)', value=current.get('body', ''), height=220, max_chars=50000)
            analysis = st.text_area('Your analysis (optional)', value=current.get('analysis', ''), max_chars=20000)
            source = st.text_input('Original source link (optional)', value=current.get('source_url', ''))
            submit = st.form_submit_button('Publish' if chosen is None else 'Save changes', type='primary')
        if submit and authorized(setting('NEWS_ADMIN_PASSWORD')):
            try:
                save_post(path, headline, published, body, analysis, source, chosen, current.get('_sha'))
            except (ValueError, StorageError) as exc:
                st.error(str(exc))
            except Exception:
                st.error('Could not save the post. Check GitHub storage configuration.')
            else:
                cached_posts.clear()
                st.rerun()
        if chosen is not None:
            with st.form(f'news_delete_{chosen}'):
                confirm = st.checkbox('I confirm I want to permanently delete this article')
                remove = st.form_submit_button('Delete article')
            if remove and authorized(setting('NEWS_ADMIN_PASSWORD')):
                if not confirm:
                    st.warning('Tick the confirmation box first.')
                else:
                    try:
                        delete_post(path, chosen, current.get('_sha'))
                    except Exception:
                        st.error('Could not delete the post. Refresh and check GitHub configuration.')
                    else:
                        cached_posts.clear()
                        st.rerun()
        st.download_button('Export news backup (JSON)', data=__import__('json').dumps(posts, ensure_ascii=False, indent=2), file_name='news_backup.json', mime='application/json')
