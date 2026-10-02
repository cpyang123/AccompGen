"""Run manually with Playwright installed: python tests/browser_editor.py."""
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
env = dict(os.environ, GRADIO_ANALYTICS_ENABLED='False', MPLCONFIGDIR='/tmp/motigen-matplotlib')
env.pop('SPACE_ID', None)
with open('/tmp/motigen-editor-server.log', 'w') as log:
    server = subprocess.Popen([sys.executable, str(ROOT / 'tests/editor_server.py')], env=env, stdout=log, stderr=log)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen('http://127.0.0.1:7869/config', timeout=1).close()
                break
            except Exception:
                if server.poll() is not None:
                    raise RuntimeError(Path('/tmp/motigen-editor-server.log').read_text())
                time.sleep(1)
        with sync_playwright() as p:
            # The synthetic cross-origin host has no real network address; allow
            # its local iframe fixture in Chromium's local-network policy.
            browser = p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-features=LocalNetworkAccessChecks'])
            page = browser.new_page(viewport={'width':1440, 'height':1100})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto('http://127.0.0.1:7869', wait_until='networkidle')
            page.wait_for_function('Boolean(window.motigenEditor)')
            get_input = lambda: page.evaluate('window.motigenEditor.getInput()')
            assert page.locator('#me-abstract select').count() == 0
            point = page.locator('[data-point="1"] .me-point').bounding_box()
            chart = page.locator('#me-contour').bounding_box()
            x,y = point['x']+point['width']/2, point['y']+point['height']/2
            page.mouse.move(x,y);page.mouse.down();page.mouse.move(x,y+100*chart['height']/280,steps=15);page.mouse.up()
            assert get_input()['abstract'] == '0,-2,3,-1', get_input()
            for _ in range(6):page.get_by_role('button', name='Add contour point', exact=True).click()
            assert len(get_input()['abstract'].split(',')) == 10
            assert page.get_by_role('button', name='Add contour point', exact=True).is_disabled()
            for _ in range(6):page.get_by_role('button', name='Remove last contour point', exact=True).click()
            assert page.get_by_role('button', name='Remove last contour point', exact=True).is_disabled()
            assert get_input()['abstract'] == '0,-2,3,-1'
            page.locator('[data-point="3"]').focus();page.keyboard.press('ArrowUp')
            assert get_input()['abstract'] == '0,-2,3,1'
            page.keyboard.press('ArrowDown')
            assert get_input()['abstract'] == '0,-2,3,-1'
            page.get_by_role('button', name='Generate music', exact=True).click()
            page.wait_for_function('Boolean(document.querySelector("#score-preview iframe")) && document.querySelector("#status").textContent.trim() === ""', timeout=30000)
            assert not page.locator('#status').is_visible()
            assert 'Selected candidate' not in page.locator('#score-panel').inner_text()
            assert page.frame_locator('iframe[title="Generated sheet music"]').locator('#legend').count() == 0
            page.get_by_text('Generation details', exact=True).click()
            assert 'Selected candidate 7/12' in page.locator('#result-details').inner_text()
            assert '0,-2,3,-1' in page.locator('.gradio-container').inner_text()
            page.get_by_text('Model prompt used', exact=True).click()
            assert '%motif:v1:step_skip_leap: 0,-2,3,-1' in page.locator('details pre').inner_text()
            page.screenshot(path=str(ROOT / 'validation/abstract-editor.png'), full_page=True)
            page.get_by_role('tab', name='Write on a staff', exact=True).click()
            assert get_input()['mode'] == 'Concrete notes'
            assert page.locator('#me-staff [data-engraver="vexflow"] .vf-stavenote').count() == 4
            assert page.locator('#me-staff [data-note] > rect').first.get_attribute('stroke') == 'none'
            assert page.evaluate("document.fonts.check('20px Bravura')")
            for button in page.locator('[data-duration]').all():
                note_box=button.locator('svg').bounding_box();button_box=button.bounding_box()
                assert note_box['y'] >= button_box['y'] and note_box['y']+note_box['height'] <= button_box['y']+button_box['height']
            assert page.locator('#me-note-list, [data-select]').count() == 0
            # Reproduce real click -> key behavior, without test-only focus().
            page.locator('[data-note="1"]').click()
            page.keyboard.press('ArrowUp')
            assert get_input()['notes'][1][0] == 'A4'
            page.keyboard.press('ArrowDown')
            assert get_input()['notes'][1][0] == 'G4'
            page.keyboard.press('ArrowRight')
            assert page.evaluate('document.activeElement.dataset.note') == '2'
            page.keyboard.press('ArrowLeft')
            assert page.evaluate('document.activeElement.dataset.note') == '1'
            assert page.locator('#me-concrete .me-help-panel').get_attribute('open') is None

            # Pointer drag from palette inserts a new pitch at the requested slot.
            staff = page.locator('#me-staff')
            def point(x,y):
                box=staff.bounding_box()
                return box['x']+x*box['width']/680, box['y']+y*box['height']/272
            source = page.get_by_role('button', name='Eighth note', exact=True).bounding_box()
            page.mouse.move(source['x']+source['width']/2, source['y']+source['height']/2)
            page.mouse.down();page.mouse.move(*point(318,120), steps=15);page.mouse.up()
            assert get_input()['notes'][-1] == ['A4','0.5'], get_input()
            # Move that note upward to C5 and left to the second slot.
            page.mouse.move(*point(318,120));page.mouse.down();page.mouse.move(*point(162,104), steps=15);page.mouse.up()
            assert get_input()['notes'][1] == ['C5','0.5'], get_input()
            page.keyboard.press('ArrowUp');page.keyboard.press('ArrowDown')
            assert get_input()['notes'][1] == ['C5','0.5']
            assert page.evaluate('document.activeElement.dataset.note') == '1'
            page.get_by_label('Accidental', exact=True).select_option('#')
            page.get_by_role('button', name='Half note', exact=True).click()
            page.get_by_label('Dotted', exact=True).check()
            assert get_input()['notes'][1] == ['C#5','3'], get_input()
            page.get_by_role('button', name='Undo', exact=True).click()
            assert get_input()['notes'][1] == ['C#5','2']
            # Keyboard and pitch selector are alternatives to dragging.
            page.locator('[data-note="1"]').focus();page.keyboard.press('ArrowDown')
            assert get_input()['notes'][1][0] == 'B#4'
            page.get_by_label('Selected note pitch').select_option('35')
            assert get_input()['notes'][1][0] == 'C#5'
            # Drag a barline into a gap, remove it, and restore it with Undo.
            before_bars = get_input()['notes']
            source = page.get_by_role('button', name='Barline tool', exact=True).bounding_box()
            page.mouse.move(source['x']+source['width']/2, source['y']+source['height']/2)
            page.mouse.down();page.mouse.move(*point(188,112), steps=15);page.mouse.up()
            assert get_input()['notes'] == before_bars[:2] + [['|','']] + before_bars[2:]
            page.locator('[data-bar-after="2"]').click()
            assert get_input()['notes'] == before_bars
            page.get_by_role('button', name='Undo', exact=True).click()
            # Selected tool also places a trailing barline with a single tap.
            page.get_by_role('button', name='Barline tool', exact=True).click()
            page.mouse.click(*point(344,112))
            with_bars = before_bars[:2] + [['|','']] + before_bars[2:] + [['|','']]
            assert get_input()['notes'] == with_bars
            page.locator('[data-bar-after="5"]').focus();page.keyboard.press('Delete')
            assert get_input()['notes'] == with_bars[:-1]
            page.get_by_role('button', name='Undo', exact=True).click()
            assert get_input()['notes'] == with_bars
            page.keyboard.press('Escape')
            # Deleting an earlier note moves its following measure division.
            page.locator('[data-note="0"]').focus();page.keyboard.press('Delete')
            assert get_input()['notes'] == with_bars[1:]
            page.get_by_role('button', name='Undo', exact=True).click()
            assert get_input()['notes'] == with_bars
            page.get_by_role('button', name='Hear motif', exact=False).click()
            page.get_by_role('button', name='Generate music', exact=True).click()
            page.wait_for_function('document.querySelector("#status").textContent.includes("Composing")', timeout=10000)
            page.wait_for_function('Boolean(document.querySelector("#score-preview iframe")) && document.querySelector("#status").textContent.trim() === ""', timeout=30000)
            assert not page.locator('#status').is_visible()
            assert 'Selected candidate' not in page.locator('#score-panel').inner_text()
            assert page.frame_locator('iframe[title="Generated sheet music"]').locator('#legend').count() == 0
            page.get_by_text('Generation details', exact=True).click()
            assert 'Selected candidate 7/12' in page.locator('#result-details').inner_text()
            frame=page.frame_locator('iframe[title="Generated sheet music"]')
            frame.locator('[data-motif-match="true"]').first.wait_for()
            page.get_by_role('tab', name='Live generation', exact=True).click()
            raw = page.get_by_label('Model output · updates while composing').input_value()
            assert '^c4' in raw, raw
            page.get_by_text('Model prompt used', exact=True).click()
            assert '%motif:abc: C2 ^c4 | G F E4 |' in page.locator('details pre').inner_text()
            page.get_by_role('tab', name='Sheet music', exact=True).click()
            page.screenshot(path=str(ROOT / 'validation/concrete-editor.png'), full_page=True)
            # Length bounds, tap-to-insert, and undo retain a valid serialized motif.
            page.get_by_role('button', name='Clear', exact=True).click()
            assert get_input()['notes'] == []
            assert 'Add 4 more' in page.locator('#me-note-status').inner_text()
            page.get_by_role('button', name='Undo', exact=True).click()
            assert get_input()['notes'] == with_bars
            for _ in range(7):
                source = page.get_by_role('button', name='Quarter note', exact=True).bounding_box()
                page.mouse.move(source['x']+source['width']/2, source['y']+source['height']/2)
                page.mouse.down();page.mouse.move(*point(350,136), steps=8);page.mouse.up()
            assert sum(row[0] != '|' for row in get_input()['notes']) == 10, get_input()
            page.get_by_role('button', name='Reset', exact=True).click()
            assert all(row[0] != '|' for row in get_input()['notes'])
            page.set_viewport_size({'width':390,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path=str(ROOT / 'validation/editor-mobile.png'), full_page=True)
            mobile = browser.new_page(viewport={'width':390,'height':844}, has_touch=True, is_mobile=True)
            mobile.on('pageerror', lambda error: errors.append(str(error)))
            mobile.goto('http://127.0.0.1:7869', wait_until='networkidle')
            # Real touch drag edits the contour without scrolling the page.
            mobile.locator('#me-contour').scroll_into_view_if_needed()
            point=mobile.locator('[data-point="1"] .me-point').bounding_box()
            chart=mobile.locator('#me-contour').bounding_box()
            x,y=point['x']+point['width']/2,point['y']+point['height']/2
            touch=mobile.context.new_cdp_session(mobile)
            touch.send('Input.dispatchTouchEvent', {'type':'touchStart','touchPoints':[{'x':x,'y':y}]})
            for i in range(1,11):
                touch.send('Input.dispatchTouchEvent', {'type':'touchMove','touchPoints':[{'x':x,'y':y+100*chart['height']/280*i/10}]})
            touch.send('Input.dispatchTouchEvent', {'type':'touchEnd','touchPoints':[]})
            assert mobile.evaluate('window.motigenEditor.getInput().abstract') == '0,-2,3,-1'
            mobile.get_by_role('tab', name='Write on a staff', exact=True).tap()
            mobile.wait_for_function('window.motigenEditor.getInput().mode === "Concrete notes"')
            mobile.get_by_role('button', name='Clear', exact=True).tap()
            mobile.locator('#me-staff').scroll_into_view_if_needed()
            for x,y in [(110,160),(162,152),(214,144),(266,128)]:
                box=mobile.locator('#me-staff').bounding_box()
                mobile.touchscreen.tap(box['x']+x*box['width']/680,box['y']+y*box['height']/272)
            assert mobile.evaluate('window.motigenEditor.getInput().notes') == [['C4','1'],['D4','1'],['E4','1'],['G4','1']]
            mobile.get_by_role('button', name='Barline tool', exact=True).tap()
            mobile.locator('#me-staff').scroll_into_view_if_needed()
            box=mobile.locator('#me-staff').bounding_box()
            mobile.touchscreen.tap(box['x']+188*box['width']/680,box['y']+112*box['height']/272)
            assert mobile.evaluate('window.motigenEditor.getInput().notes') == [['C4','1'],['D4','1'],['|',''],['E4','1'],['G4','1']]
            mobile.locator('[data-bar-after="2"]').tap()
            assert len(mobile.evaluate('window.motigenEditor.getInput().notes')) == 4
            mobile.get_by_role('button', name='Scroll staff right', exact=True).tap()
            mobile.wait_for_function('document.querySelector(".me-staff-scroll").scrollLeft > 100')
            mobile.close()
            # Advance only the cat's timer, preserving normal app/browser timers.
            cat_page=browser.new_page(viewport={'width':390,'height':844})
            cat_page.add_init_script("""window.__catTicks=[]; const realInterval=window.setInterval;
                window.setInterval=(fn,ms,...args)=>{if(ms===180000){window.__catTicks.push(fn);return 8675309;}return realInterval(fn,ms,...args);};""")
            cat_page.goto('http://127.0.0.1:7869',wait_until='networkidle')
            cat_page.wait_for_function('Boolean(window.motigenEditor)')
            assert cat_page.evaluate('window.__catTicks.length') == 1
            assert cat_page.locator('.cat-walker').count() == 0
            cat_page.evaluate('window.__catTicks[0]()')
            cat_page.wait_for_function('document.querySelector("#motigen-cat-lane").style.visibility === "visible"')
            assert abs(cat_page.locator('#motigen-cat-lane').bounding_box()['y'] + 40 - 844) < 2
            cat_page.evaluate('window.scrollTo(0,300)')
            assert abs(cat_page.locator('#motigen-cat-lane').bounding_box()['y'] + 40 - 844) < 2
            assert cat_page.locator('.cat-walker.ltr').count() == 1
            cat_page.evaluate('window.__catTicks[0]()')
            assert cat_page.locator('.cat-walker').count() == 1
            assert cat_page.locator('#motigen-cat-lane').evaluate("el=>getComputedStyle(el).pointerEvents") == 'none'
            assert cat_page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            cat_page.evaluate("document.querySelector('.cat-walker').dispatchEvent(new Event('animationend'))")
            assert cat_page.locator('.cat-walker').count() == 0
            cat_page.evaluate('window.__catTicks[0]()')
            assert cat_page.locator('.cat-walker.rtl').count() == 1
            cat_page.locator('.cat-walker').evaluate('el=>el.remove()')
            cat_page.emulate_media(reduced_motion='reduce')
            cat_page.evaluate('window.__catTicks[0]()')
            assert cat_page.locator('.cat-walker').count() == 0
            cat_page.close()
            # Hugging Face can embed the app in a frame taller than the screen.
            # Use different hostnames so this also exercises cross-origin clipping.
            embedded=browser.new_page(viewport={'width':1000,'height':800})
            embedded.on('pageerror', lambda error: errors.append(str(error)))
            embedded.route('http://localhost:7869/embed-test', lambda route: route.fulfill(content_type='text/html',body='<body style="margin:0"><div style="height:180px">Host header</div><iframe src="http://127.0.0.1:7869" style="border:0;width:100%;height:2400px"></iframe><div style="height:600px">Host footer</div></body>'))
            embedded.goto('http://localhost:7869/embed-test',wait_until='networkidle')
            embedded.frame_locator('iframe').locator('#motif-editor').wait_for()
            child=embedded.locator('iframe').element_handle().content_frame()
            child.wait_for_function('Boolean(window.walkCat)')
            child.evaluate('window.walkCat()')
            child.wait_for_function('document.querySelector("#motigen-cat-lane").style.visibility === "visible"')
            def cat_at_bottom():
                box=child.locator('#motigen-cat-lane').bounding_box()
                return abs(box['y']+box['height']-embedded.viewport_size['height'])<2
            assert cat_at_bottom()
            for scroll in (350,850):
                embedded.evaluate('(y)=>window.scrollTo(0,y)',scroll)
                for _ in range(20):
                    if cat_at_bottom():break
                    embedded.wait_for_timeout(100)
                assert cat_at_bottom(), child.locator('#motigen-cat-lane').bounding_box()
            embedded.set_viewport_size({'width':390,'height':700})
            for _ in range(20):
                if cat_at_bottom():break
                embedded.wait_for_timeout(100)
            assert cat_at_bottom()
            embedded.screenshot(path=str(ROOT/'validation/cat-embedded-viewport.png'))
            embedded.close()

            assert not errors, errors
            print('PASS: click/drag then keyboard note editing, viewport cat in cross-origin embed while scrolling/resizing, draggable contour (mouse/touch/keyboard), three-minute cat scheduling/cleanup, no duplicate note chips, 4–10 limits, barline drag/tap/remove/undo/prompt, palette drag, pitch drag, reorder, durations, accidental, dotted note, keyboard, undo, clear, preview audio, both generation payloads, highlighted result, mobile layout', flush=True)
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=15)
