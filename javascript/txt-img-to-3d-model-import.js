(function () {
    if (!globalThis.txtImgTo3DModel) globalThis.txtImgTo3DModel = {};
    const txtImgTo3DModel = globalThis.txtImgTo3DModel;

    function load(cont) {
        const scripts = cont.textContent.trim().split('\n');
        const base_path = `/file=${scripts.shift()}/js`;
        cont.textContent = '';

        const df = document.createDocumentFragment();
        for (let src of scripts) {
            const script = document.createElement('script');
            script.async = true;
            script.type = 'module';
            script.src = `file=${src}`;
            df.appendChild(script);
        }

        if (!globalThis.txtImgTo3DModel.imports) {
            globalThis.txtImgTo3DModel.imports = {};
        }

        cont.appendChild(df);


    }

    onUiLoaded(function () {
        txtImgTo3DModelImport = gradioApp().querySelector('#txt-img-to-3d-model-import');
        load(txtImgTo3DModelImport);
    });
})();