import React from 'react';
import { StyleSheet, Text, WebView } from 'react-native';


export default class App extends React.Component {
  render() {
    return (
      <WebView
        source={{uri: 'https://www.google.com/maps/d/u/0/embed?mid=1biMHnY5kx9KgncoUlp9xESeOHIE'}}
        style={{marginTop: 20}}>
      </WebView>



    );
  }
}
