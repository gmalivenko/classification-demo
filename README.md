# Tensorflow.js classification demo application

## vue.js project setup

```
npm install
```

### Compiles and hot-reloads for development
```
npm run serve
```

### Compiles and minifies for production
```
npm run build
```

### Run your tests
```
npm run test
```

### Lints and fixes files
```
npm run lint
```

## Convert models

Please look into the `convert` directory. It contains small snippet script that allow to download model and convert it to `frozen` tensorflow model format.
Then, you can convert it to `Tensorflow.js` format using the following command:

```
tensorflowjs_converter --input_format=tf_frozen_model saved_model.pb TF_JS_DESTINATION --output_node_names OUTOUT_OP_NAME --saved_model_tags=serve --output_json=True
```


## License
This software is covered by MIT License.