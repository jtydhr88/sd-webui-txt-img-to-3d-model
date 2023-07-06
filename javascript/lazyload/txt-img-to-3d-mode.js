console.log('[Txt/Img to 3D Model] loading...');

async function _import() {

}

await _import();

let _r = 0;

(async function () {
    let container = document.getElementById('txt-img-to-3d-model-container');

    let select = document.createElement('select');

    select.id = "mySelect";

    container.appendChild(select);

    let refreshButton = document.createElement('button');

    refreshButton.innerText = "Refresh";

    refreshButton.addEventListener('click', async function () {
        while (select.firstChild) {
            select.removeChild(select.firstChild);
        }

        const getGeneratedHistory = await py2js('getGeneratedHistory');

        let arr = JSON.parse(getGeneratedHistory.replace(/'/g, '"'));

        for (let i = 0; i < arr.length; i++) {
            let option = document.createElement('option');
            option.value = arr[i];
            option.text = arr[i];
            select.appendChild(option);
        }
    });

    container.appendChild(refreshButton);

    refreshButton.click();

    container.appendChild(document.createElement('br'));

    const tabsDiv = document.getElementById('tabs');

    const tabNavDiv = tabsDiv.getElementsByClassName('tab-nav')[0];

    const buttons = tabNavDiv.getElementsByTagName('button');

    let editorButton;

    for (let i = 0; i < buttons.length; i++) {
        if (buttons[i].innerText === '3D Editor') {
            editorButton = buttons[i];
            break;
        }
    }

    if (editorButton) {
        let button = document.createElement('button');

        button.innerText = "Open In 3D Editor";

        button.addEventListener('click', function () {
            let selectedValue = document.getElementById('mySelect').value;

            window.threeDEditorImportFileFromUrl('/file=extensions/sd-webui-txt-img-to-3d-model/outputs/' + selectedValue);

            alert("please open 3D editor tab");
        });

        container.appendChild(button);
    } else {
        let a = document.createElement('a');

        a.href = 'https://github.com/jtydhr88/sd-webui-3d-editor';

        let linkText = document.createTextNode('here');

        a.appendChild(linkText);

        let text1 = document.createTextNode('No 3D Editor extension found, please go to ');
        let text2 = document.createTextNode(' to install.');

        container.appendChild(text1);
        container.appendChild(a);
        container.appendChild(text2);
    }


    function to_gradio(v) {
        return [v, _r++];
    }

    function py2js(pyname, ...args) {
        // call python's function
        // (1) Set args to gradio's field
        // (2) Click gradio's button
        // (3) JS callback will be kicked with return value from gradio

        // (1)
        return (args.length === 0 ? Promise.resolve() : js2py(pyname + '_args', JSON.stringify(args)))
            .then(() => {
                return new Promise(resolve => {
                    const callback_name = `txt-img-to-3d-model-${pyname}`;
                    // (3)
                    globalThis[callback_name] = value => {
                        delete globalThis[callback_name];
                        resolve(value);
                    }
                    // (2)
                    gradioApp().querySelector(`#${callback_name}_get`).click();
                });
            });
    }


    function js2py(gradio_field, value) {
        return new Promise(resolve => {
            const callback_name = `txt-img-to-3d-model-${gradio_field}`;

            // (2)
            globalThis[callback_name] = () => {

                delete globalThis[callback_name];

                // (3)
                const callback_after = callback_name + '_after';
                globalThis[callback_after] = () => {
                    delete globalThis[callback_after];
                    resolve();
                };

                return to_gradio(value);
            };

            // (1)
            gradioApp().querySelector(`#${callback_name}_set`).click();
        });
    }
})();